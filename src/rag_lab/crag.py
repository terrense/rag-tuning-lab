"""
================================================================================
crag.py —— CRAG-lite：检索自评 + 纠错（Corrective RAG 的轻量版）
--------------------------------------------------------------------------------
普通 RAG 的一个致命失败：检索到一堆不相关的块，生成器还是硬编一个答案出来
（幻觉）。CRAG 的思路：**检索完先自我评估质量，再决定怎么办**。

三档决策（对应 CRAG 论文的 Correct / Ambiguous / Incorrect）：
  · 高置信（有明显相关块）        → proceed：照常生成
  · 低置信（全都不相关）          → abstain：诚实拒答，不硬编
  · 中间（相关但不足）            → refine：改写 query 重检一次，取更好的一组

评估器用便宜的 deepseek-flash（llm.roles 里的 crag 角色，默认 flash）：一次调用
给 top-k 每个块打相关性分。便宜、够用——这正是"轻任务用便宜模型"的落点。

怎么证明它有用（eval 模式）：v2 评测集有 expected_source_ids，于是能把题分成
  · 检索成功（金标文档在 top-k）：CRAG 不该误拒答（false abstain 是坏事）
  · 检索失败（金标不在 top-k）  ：CRAG 该拒答 or 靠改写把金标救回来
指标：abstain 正确率、改写救回率、误拒答率。对比普通 RAG"失败也硬编"。

用法：
  python -m rag_lab.crag --config configs/diseases.yaml --query "..."         # 单问演示
  python -m rag_lab.crag --config configs/diseases.yaml --eval --limit 40      # 评测
================================================================================
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from rag_lab.llm import chat
from rag_lab.models import SearchHit

_GRADE_SYS = (
    "你是检索质量评估器。给你一个【问题】和若干【候选资料】，逐条判断该资料是否"
    "真正有助于回答这个问题。只看相关性，不要自己补充知识。\n"
    "对每条给一个 0-1 的相关性分：1=直接回答该问题的核心；0.5=沾边但不充分；"
    "0=无关。只输出 JSON 数组：[{\"i\": 序号, \"score\": 分数}]，不要其它文字。"
)


def grade_hits(cfg: dict[str, Any], query: str, hits: list[SearchHit],
               k: int = 5, trace: Any = None) -> list[float]:
    """一次 flash 调用给 top-k 每个块打相关性分。返回分数列表（对齐 hits[:k]）。"""
    use = hits[:k]
    if not use:
        return []
    blocks = []
    for i, h in enumerate(use):
        title = h.title or h.source_id
        blocks.append(f"[{i}] {title}\n{(h.text or '')[:280]}")
    user = f"【问题】{query}\n\n【候选资料】\n" + "\n\n".join(blocks)
    out = chat(cfg, [{"role": "system", "content": _GRADE_SYS},
                     {"role": "user", "content": user}],
               role="crag", max_tokens=400, temperature=0.0, trace=trace)
    scores = [0.0] * len(use)
    s, e = out["text"].find("["), out["text"].rfind("]")
    if s >= 0 and e > s:
        try:
            for row in json.loads(out["text"][s:e + 1]):
                i = int(row.get("i"))
                if 0 <= i < len(use):
                    scores[i] = max(0.0, min(1.0, float(row.get("score", 0.0))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return scores


def crag_retrieve(cfg: dict[str, Any], query: str, history: list[str] | None = None,
                  trace: Any = None) -> dict[str, Any]:
    """带自纠错的检索。返回 {hits, decision, top_score, refined, scores}。

    decision ∈ {proceed, refined, abstain}。阈值可配 crag.high / crag.low。
    """
    from rag_lab.pipeline import query_config

    crag_cfg = cfg.get("crag", {})
    high = float(crag_cfg.get("high", 0.6))    # 最高分 ≥ high → 直接用
    low = float(crag_cfg.get("low", 0.3))      # 最高分 < low → 拒答
    k = int(crag_cfg.get("grade_k", 5))

    result = query_config(cfg, query, history=history)
    hits = result["hits"]
    scores = grade_hits(cfg, query, hits, k=k, trace=trace)
    top = max(scores) if scores else 0.0

    if top >= high:
        return {"hits": hits, "decision": "proceed", "top_score": top,
                "refined": False, "scores": scores}

    if top < low:
        # 全都不相关 → 先试一次改写重检；改写后仍差才拒答
        from rag_lab.query_rewrite import layer3_rewrite
        rq = layer3_rewrite(cfg, query, history)
        if rq and rq != query:
            r2 = query_config(cfg, rq, history=history)
            s2 = grade_hits(cfg, rq, r2["hits"], k=k, trace=trace)
            if (max(s2) if s2 else 0.0) >= low:
                return {"hits": r2["hits"], "decision": "refined", "top_score": max(s2),
                        "refined": True, "rewrite": rq, "scores": s2}
        return {"hits": [], "decision": "abstain", "top_score": top,
                "refined": False, "scores": scores}

    # 中间档：改写重检，取相关性更高的一组
    from rag_lab.query_rewrite import layer3_rewrite
    rq = layer3_rewrite(cfg, query, history)
    if rq and rq != query:
        r2 = query_config(cfg, rq, history=history)
        s2 = grade_hits(cfg, rq, r2["hits"], k=k, trace=trace)
        if (max(s2) if s2 else 0.0) > top:
            return {"hits": r2["hits"], "decision": "refined", "top_score": max(s2),
                    "refined": True, "rewrite": rq, "scores": s2}
    return {"hits": hits, "decision": "proceed", "top_score": top,
            "refined": False, "scores": scores}


# ---------------------------------------------------------------------------
# 评测：CRAG 的决策对不对（用 expected_source_ids 当真值）
# ---------------------------------------------------------------------------
def evaluate(cfg: dict[str, Any], limit: int = 0) -> dict[str, Any]:
    from rag_lab.config import get_path
    from rag_lab.loaders import load_eval_queries

    queries = load_eval_queries(get_path(cfg, "eval_queries"))
    if limit:
        queries = queries[:limit]

    # 混淆矩阵：检索是否成功(金标是否在原始 top-k) × CRAG 是否拒答
    n = 0
    retrieved_ok = 0           # 原始检索成功数
    correct_abstain = 0        # 检索失败 且 CRAG 拒答（对）
    should_abstain = 0         # 检索失败总数
    false_abstain = 0          # 检索成功 却 CRAG 拒答（错）
    recovered = 0              # 原始失败，改写后金标回到 top-k（救回）
    refined_total = 0
    rows = []

    from rag_lab.pipeline import query_config
    for item in queries:
        q = str(item["question"])
        expected = set(item.get("expected_source_ids", []))
        base = query_config(cfg, q)
        base_ids = [h.source_id for h in base["hits"][:5]]
        base_hit = bool(expected & set(base_ids))
        retrieved_ok += int(base_hit)

        out = crag_retrieve(cfg, q)
        dec = out["decision"]
        refined_total += int(out["refined"])
        final_ids = [h.source_id for h in out["hits"][:5]]

        if not base_hit:
            should_abstain += 1
            if dec == "abstain":
                correct_abstain += 1
            if out["refined"] and (expected & set(final_ids)):
                recovered += 1        # 改写把金标救回来了
        else:
            if dec == "abstain":
                false_abstain += 1
        n += 1
        rows.append({"id": item.get("id"), "base_hit": base_hit,
                     "decision": dec, "top_score": round(out["top_score"], 2)})

    return {
        "n": n, "retrieved_ok": retrieved_ok, "should_abstain": should_abstain,
        "correct_abstain": correct_abstain, "false_abstain": false_abstain,
        "recovered": recovered, "refined_total": refined_total,
        "abstain_recall": correct_abstain / should_abstain if should_abstain else 0.0,
        "false_abstain_rate": false_abstain / retrieved_ok if retrieved_ok else 0.0,
        "rows": rows,
    }


def main() -> None:
    from rag_lab.config import load_config, parse_value, set_dotted

    ap = argparse.ArgumentParser(description="CRAG-lite: retrieval self-grading + correction.")
    ap.add_argument("--config", default="configs/diseases.yaml")
    ap.add_argument("--query", default="")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        key, raw = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw))

    if args.eval:
        r = evaluate(cfg, args.limit)
        print(f"\nCRAG eval (N={r['n']}):")
        print(f"  原始检索成功 {r['retrieved_ok']}/{r['n']} | 检索失败 {r['should_abstain']}")
        print(f"  正确拒答 {r['correct_abstain']}/{r['should_abstain']} (abstain_recall={r['abstain_recall']:.2f})")
        print(f"  改写救回金标 {r['recovered']} | 触发改写 {r['refined_total']}")
        print(f"  误拒答 {r['false_abstain']}/{r['retrieved_ok']} (false_abstain_rate={r['false_abstain_rate']:.2f})")
    elif args.query:
        out = crag_retrieve(cfg, args.query)
        print(f"decision={out['decision']} top_score={out['top_score']:.2f} refined={out['refined']}")
        if out.get("rewrite"):
            print(f"rewrite: {out['rewrite']}")
        for h in out["hits"][:5]:
            print(f"  {h.source_id}  {h.title}")
        if out["decision"] == "abstain":
            print("  → 根据现有资料无法确定（诚实拒答，不硬编）")
    else:
        ap.error("give --query or --eval")


if __name__ == "__main__":
    main()
