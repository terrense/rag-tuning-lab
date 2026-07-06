"""
================================================================================
evalgen.py —— P0-E0：自动扩建评测集（LLM 出题 + 双模型交叉质检）
--------------------------------------------------------------------------------
动机：10 题的评测集上 0.70 vs 0.60 毫无统计意义。要把 LEADERBOARD 上的每个
结论变成"可信的数字"，评测集必须上到百题量级——手写不现实，所以让 LLM 出题。

流水线（每一步都在防"垃圾进、垃圾出"）：

  1) 分层采样   按就诊科室对 5942 条疾病记录分层，按比例抽 N 条
                （不分层会被内科/皮肤科这种大科室淹没，小科室零覆盖）。
  2) flash 出题 deepseek-flash 批量生成，四种风格轮转：
                  symptom_led   症状导向，不许出现病名（考语义检索）
                  name_led      病名导向（考关键词/BM25）
                  exam_treat    检查/治疗导向（考字段召回）
                  colloquial    患者口语（考 query 改写层，如"拉肚子"→"腹泻"）
  3) 规则校验   程序硬检查：症状导向/口语题不得泄漏病名、长度合理、去重。
  4) pro 质检   deepseek-pro 逐题判：该记录能否回答此题？症状组合是否
                足够指向该病（区分度）？——出题者和质检者是两个模型，
                单模型的系统性盲区不会既骗过出题又骗过质检。
  5) 落盘       data/eval_queries_diseases_v2.yaml，带风格/来源标注；
                手写的 10 题作为质量锚点合并在最前面。

断点续跑：中间结果缓存到 storage/evalgen_cache.json，重跑不重复计费。

用法：
    python -m rag_lab.evalgen --n 150 --seed 42
    python -m rag_lab.evalgen --n 150 --seed 42 --fresh   # 忽略缓存重来
================================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_lab.llm import chat
from rag_lab.structured import load_disease_json

CACHE_PATH = Path("storage/evalgen_cache.json")
OUT_PATH = Path("data/eval_queries_diseases_v2.yaml")
MANUAL_PATH = Path("data/eval_queries_diseases.yaml")
DATA_PATH = Path("data/files/数据集/diseases_clean.json")

STYLES = ["symptom_led", "name_led", "exam_treat", "colloquial"]

STYLE_RULES = {
    "symptom_led": "症状导向：病人描述若干典型症状问'可能是什么病/怎么办'。绝对不能出现疾病名称本身。挑该病最有区分度的症状组合（能把它和常见相似病区分开）。",
    "name_led": "病名导向：直接围绕疾病名称提问（症状/病因/治疗/挂哪个科等某一方面）。",
    "exam_treat": "检查治疗导向：围绕该病的检查项目或治疗方式提问，可含病名，例如'确诊X需要做哪些检查''X一般怎么治'。",
    "colloquial": "患者口语：用大白话描述症状（如'拉肚子''心慌''没劲'这类口语词，不用医学术语），语气像普通人在网上问病。绝对不能出现疾病名称本身。",
}

_GEN_SYS = (
    "你是医疗检索评测集的出题人。给你若干条疾病记录，每条指定了出题风格，"
    "请为每条各出一道中文问题。要求：\n"
    "- 问题必须仅凭该条记录就能回答（不要问记录里没有的信息）；\n"
    "- 15~45 个字，自然、像真实用户会问的；\n"
    "- 严格遵守各条的风格要求，尤其'不能出现疾病名称'的约束；\n"
    "- 只输出 JSON 数组：[{\"index\": 序号, \"question\": \"问题\"}]，不要其它文字。"
)

_FILTER_SYS = (
    "你是医疗检索评测集的质检员。给你若干道题，每道附带它对应的疾病记录摘要。"
    "对每道题独立判断：\n"
    "a) answerable：仅凭该记录能否回答该问题；\n"
    "b) specific：若问题不含病名（症状导向题），其症状组合是否足够指向该病"
    "（若这些症状同样典型地指向许多其它常见病，算不合格）；含病名的题此项恒为 true；\n"
    "c) natural：是否像真实用户的自然提问。\n"
    "三项都合格 keep=true，否则 keep=false 并给一句话理由。\n"
    "只输出 JSON 数组：[{\"index\": 序号, \"keep\": true/false, \"reason\": \"...\"}]。"
)


def _record_brief(doc: dict[str, Any], with_name: bool = True) -> str:
    """把一条疾病文档压成给 LLM 看的简介（控制 token）。"""
    m = doc["metadata"]
    parts = []
    if with_name:
        parts.append(f"疾病名称：{m['disease_name']}")
    if m.get("department"):
        parts.append(f"就诊科室：{m['department']}")
    if m.get("symptoms"):
        parts.append(f"症状：{m['symptoms']}")
    if m.get("exams"):
        parts.append(f"检查项目：{m['exams']}")
    if m.get("treatments"):
        parts.append(f"治疗方式：{m['treatments']}")
    desc = doc.get("content", "")
    i = desc.find("【描述】")
    if i >= 0:
        parts.append("描述节选：" + desc[i + 4 : i + 200].strip().replace("\n", " "))
    return "\n".join(parts)


def _parse_json_array(text: str) -> list[dict]:
    """从 LLM 输出里抠出 JSON 数组（容忍 ```json 围栏和前后废话）。"""
    s, e = text.find("["), text.rfind("]")
    if s < 0 or e <= s:
        return []
    try:
        out = json.loads(text[s : e + 1])
        return out if isinstance(out, list) else []
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# 1) 分层采样：按科室比例抽样，保证小科室也有覆盖
# ---------------------------------------------------------------------------
def stratified_sample(docs: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_dept: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        dept = (d["metadata"].get("department") or "未知").split("、")[0]
        by_dept[dept].append(d)
    total = len(docs)
    picked: list[dict] = []
    # 每个科室按占比分名额（四舍五入，至少 1），再全局裁到 n
    for dept, group in sorted(by_dept.items(), key=lambda kv: -len(kv[1])):
        quota = max(1, round(n * len(group) / total))
        rng.shuffle(group)
        picked.extend(group[:quota])
    rng.shuffle(picked)
    return picked[:n]


# ---------------------------------------------------------------------------
# 2) flash 批量出题（每批 5 条，带断点缓存）
# ---------------------------------------------------------------------------
def generate_questions(cfg: dict, sample: list[dict], cache: dict) -> list[dict]:
    items = cache.setdefault("items", {})   # doc_id -> {question, style, ...}
    todo = [(i, d) for i, d in enumerate(sample) if d["id"] not in items]
    print(f"[gen] total={len(sample)} cached={len(sample) - len(todo)} todo={len(todo)}")
    BATCH = 5
    for b in range(0, len(todo), BATCH):
        batch = todo[b : b + BATCH]
        lines = []
        for j, (i, d) in enumerate(batch):
            style = STYLES[i % len(STYLES)]
            hide = style in ("symptom_led", "colloquial")
            lines.append(
                f"### 第{j}条（风格：{style}）\n要求：{STYLE_RULES[style]}\n"
                + _record_brief(d, with_name=True)
                + ("\n（提醒：这条的问题里不能出现上面的疾病名称）" if hide else "")
            )
        out = chat(cfg, [{"role": "system", "content": _GEN_SYS},
                         {"role": "user", "content": "\n\n".join(lines)}],
                   role="evalgen", max_tokens=1500, temperature=0.7)
        rows = _parse_json_array(out["text"])
        got = 0
        for row in rows:
            try:
                j = int(row.get("index"))
                q = str(row.get("question", "")).strip()
            except (TypeError, ValueError):
                continue
            if 0 <= j < len(batch) and q:
                i, d = batch[j]
                items[d["id"]] = {"question": q, "style": STYLES[i % len(STYLES)],
                                  "disease": d["metadata"]["disease_name"]}
                got += 1
        _save_cache(cache)
        print(f"[gen] batch {b // BATCH + 1}/{(len(todo) + BATCH - 1) // BATCH}: +{got}")
    return [dict(id=k, **v) for k, v in items.items()]


# ---------------------------------------------------------------------------
# 3) 程序规则校验 + 去重
# ---------------------------------------------------------------------------
def _bigrams(s: str) -> set[str]:
    s = re.sub(r"[^\w一-鿿]", "", s)
    return {s[i : i + 2] for i in range(len(s) - 1)}


def rule_check(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    seen: list[tuple[set[str], str]] = []
    for r in rows:
        q, style, disease = r["question"], r["style"], r["disease"]
        reason = None
        if not (10 <= len(q) <= 60):
            reason = f"长度{len(q)}越界"
        elif style in ("symptom_led", "colloquial") and disease and disease in q:
            reason = "泄漏病名"
        else:
            bg = _bigrams(q)
            for other_bg, other_id in seen:
                inter = len(bg & other_bg)
                if bg and inter / max(1, min(len(bg), len(other_bg))) > 0.8:
                    reason = f"与{other_id}近重复"
                    break
        if reason:
            r["drop_reason"] = f"rule:{reason}"
            dropped.append(r)
        else:
            seen.append((_bigrams(q), r["id"]))
            kept.append(r)
    return kept, dropped


# ---------------------------------------------------------------------------
# 4) pro 质检（每批 8 条，带断点缓存）
# ---------------------------------------------------------------------------
def llm_filter(cfg: dict, rows: list[dict], doc_by_id: dict[str, dict], cache: dict) -> None:
    verdicts = cache.setdefault("verdicts", {})   # doc_id -> {keep, reason}
    todo = [r for r in rows if r["id"] not in verdicts]
    print(f"[filter] total={len(rows)} cached={len(rows) - len(todo)} todo={len(todo)}")
    BATCH = 5
    for b in range(0, len(todo), BATCH):
        batch = todo[b : b + BATCH]
        lines = []
        for j, r in enumerate(batch):
            lines.append(f"### 第{j}题\n问题：{r['question']}\n对应记录：\n"
                         + _record_brief(doc_by_id[r["id"]], with_name=True))
        messages = [{"role": "system", "content": _FILTER_SYS},
                    {"role": "user", "content": "\n\n".join(lines)}]
        out = chat(cfg, messages, role="filter", max_tokens=3000, temperature=0.0)
        rows_parsed = _parse_json_array(out["text"])
        if not rows_parsed:                    # 解析失败 ≠ 题不合格：重试一次，别误杀
            print(f"[filter] parse failed, retrying. head={out['text'][:120]!r}")
            out = chat(cfg, messages, role="filter", max_tokens=4000, temperature=0.0)
            rows_parsed = _parse_json_array(out["text"])
        for row in rows_parsed:
            try:
                j = int(row.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= j < len(batch):
                verdicts[batch[j]["id"]] = {"keep": bool(row.get("keep")),
                                            "reason": str(row.get("reason", ""))[:80]}
        _save_cache(cache)
        kept_so_far = sum(1 for v in verdicts.values() if v["keep"])
        print(f"[filter] batch {b // BATCH + 1}/{(len(todo) + BATCH - 1) // BATCH}: kept={kept_so_far}/{len(verdicts)}")


# ---------------------------------------------------------------------------
# 5) 输出 YAML（手写 10 题在前作为锚点，生成题在后）
# ---------------------------------------------------------------------------
def write_yaml(final: list[dict], meta: dict) -> None:
    import yaml

    manual_text = ""
    if MANUAL_PATH.exists():
        manual = yaml.safe_load(MANUAL_PATH.read_text(encoding="utf-8")) or {}
        manual_rows = manual.get("queries", [])
        manual_text = yaml.safe_dump({"queries": manual_rows}, allow_unicode=True,
                                     sort_keys=False, width=120)
        # 去掉外层 "queries:" 行，后面统一拼
        manual_text = "\n".join(manual_text.splitlines()[1:]) + "\n"

    gen_rows = []
    for r in final:
        gen_rows.append({
            "id": f"g_{r['id'].split('_')[-1]}_{r['style']}",
            "question": r["question"],
            "expected_source_ids": [r["id"]],
            "note": f"[auto:{r['style']}] {r['disease']}",
        })
    gen_text = yaml.safe_dump({"queries": gen_rows}, allow_unicode=True,
                              sort_keys=False, width=120)
    gen_text = "\n".join(gen_text.splitlines()[1:]) + "\n"

    header = (
        "# 医疗评测集 v2 —— 手写锚点(10题) + LLM 自动生成\n"
        f"# 生成: {meta['generator']} | 质检: {meta['filter']} | seed={meta['seed']}\n"
        f"# 流程: 科室分层采样 -> flash 出题(四风格轮转) -> 规则校验 -> pro 质检\n"
        f"# 统计: 采样{meta['sampled']} 生成{meta['generated']} 规则淘汰{meta['rule_dropped']} "
        f"质检淘汰{meta['llm_dropped']} 最终{meta['final']} (+10 手写)\n"
        "# 由 python -m rag_lab.evalgen 生成；不要手改生成题，重跑脚本即可。\n"
    )
    OUT_PATH.write_text(header + "queries:\n" + manual_text + gen_text, encoding="utf-8")
    print(f"[out] {OUT_PATH}  manual=10 generated={len(gen_rows)}")


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-build the v2 disease eval set.")
    ap.add_argument("--n", type=int, default=150, help="采样多少条疾病记录（过滤后会略少）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true", help="忽略断点缓存，从头重来")
    args = ap.parse_args()

    cfg: dict = {}   # evalgen/filter 角色走 DEFAULT_ROLES：flash 出题、pro 质检
    docs = load_disease_json(DATA_PATH)
    doc_by_id = {d["id"]: d for d in docs}
    print(f"[load] {len(docs)} disease docs")

    cache: dict = {}
    if CACHE_PATH.exists() and not args.fresh:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if cache.get("seed") != args.seed or cache.get("n") != args.n:
            print("[cache] seed/n 变了，丢弃旧缓存")
            cache = {}
    cache.setdefault("seed", args.seed)
    cache.setdefault("n", args.n)

    sample = stratified_sample(docs, args.n, args.seed)
    rows = generate_questions(cfg, sample, cache)
    kept, rule_dropped = rule_check(rows)
    llm_filter(cfg, kept, doc_by_id, cache)
    verdicts = cache["verdicts"]
    final = [r for r in kept if verdicts.get(r["id"], {}).get("keep")]
    llm_dropped = [r for r in kept if not verdicts.get(r["id"], {}).get("keep")]

    from rag_lab.llm import resolve_role
    write_yaml(final, {
        "generator": resolve_role(cfg, "evalgen"), "filter": resolve_role(cfg, "filter"),
        "seed": args.seed, "sampled": len(sample), "generated": len(rows),
        "rule_dropped": len(rule_dropped), "llm_dropped": len(llm_dropped),
        "final": len(final),
    })
    for r in (rule_dropped + llm_dropped)[:10]:
        why = r.get("drop_reason") or verdicts.get(r["id"], {}).get("reason", "")
        print(f"  [dropped] {r['disease']} | {r['question'][:30]} | {why}")


if __name__ == "__main__":
    main()
