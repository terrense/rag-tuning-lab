"""
================================================================================
graph_extract.py —— GraphRAG 第1步：从文本抽取 (实体→关系→实体) 三元组
--------------------------------------------------------------------------------
普通 RAG 把文档当"孤立文本块"；GraphRAG 先把文档变成"关系网"。这一步就是
关系网的原料：让 MiniMax 读一段文本，抽出结构化的三元组，例如：
  {"head": "MaPLe", "relation": "基于", "tail": "CLIP"}
  {"head": "MaPLe", "relation": "提出", "tail": "多模态提示学习"}

后续步骤会把这些三元组拼成图、做检索。本步只负责"抽"。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rag_lab.config import load_config
from rag_lab.generate import call_minimax

TRIPLES_FILE = Path("storage/graph_triples.json")

_SYS = (
    "你是知识图谱抽取器。从给定的学术/技术文本中抽取实体之间的关系，"
    "输出三元组。要求：\n"
    "1) 实体用规范简洁的名词（方法名、模型、任务、数据集、概念、机构等），"
    "同一实体不同写法请归一（如 CLIP 与 clip-ViT-B-32 视作 CLIP）。\n"
    "2) 关系用简短中文动词短语（如 提出、基于、用于、改进、包含、对比、属于）。\n"
    "3) 只抽文本明确支持的关系，不要臆造。\n"
    '4) 只输出 JSON 数组，每项 {"head":"", "relation":"", "tail":""}，不要解释。'
)


def _parse_json_array(text: str) -> list[dict]:
    """从模型输出里稳健地抠出 JSON 数组（容忍 ```json 包裹等）。"""
    s = text.find("[")
    e = text.rfind("]")
    if s == -1 or e == -1 or e < s:
        return []
    try:
        data = json.loads(text[s : e + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for d in data:
        if isinstance(d, dict) and d.get("head") and d.get("relation") and d.get("tail"):
            out.append({"head": str(d["head"]).strip(),
                        "relation": str(d["relation"]).strip(),
                        "tail": str(d["tail"]).strip()})
    return out


def extract_triples(cfg: dict, text: str, max_triples: int = 12) -> list[dict]:
    """对一段文本抽三元组。"""
    user = f"最多抽 {max_triples} 个三元组。文本：\n{text[:1500]}"
    out = call_minimax(cfg, [{"role": "system", "content": _SYS},
                             {"role": "user", "content": user}], max_tokens=800)
    return _parse_json_array(out["text"])[:max_triples]


def build_sample(cfg: dict, keywords: list[str], per_paper: int) -> dict:
    """每个关键词（论文）各取 per_paper 个文本块抽三元组，保证跨论文覆盖。"""
    chunks_path = cfg["paths"]["chunks_cache"]
    picked = []
    counts = {k: 0 for k in keywords}
    for line in open(chunks_path, encoding="utf-8"):
        o = json.loads(line)
        m = o["metadata"]
        sid = m.get("source_id", "") or ""
        if m.get("modality") != "text":
            continue
        for k in keywords:                       # 命中哪篇就计入哪篇的配额
            if k in sid and counts[k] < per_paper:
                picked.append(o)
                counts[k] += 1
                break
        if all(c >= per_paper for c in counts.values()):
            break

    all_triples = []
    for o in picked:
        tris = extract_triples(cfg, o["text"])
        for t in tris:
            t["source_id"] = o["metadata"].get("source_id")
        all_triples.extend(tris)
        print(f"  [{o['metadata'].get('source_id','')[:48]}] 抽到 {len(tris)} 个")

    # 简单统计
    entities = set()
    for t in all_triples:
        entities.add(t["head"]); entities.add(t["tail"])
    result = {"triples": all_triples, "num_triples": len(all_triples),
              "num_entities": len(entities), "num_chunks": len(picked)}
    TRIPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRIPLES_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="GraphRAG step 1: extract triples.")
    p.add_argument("--config", default="configs/docs.yaml")
    p.add_argument("--keywords", default="maple,graph_neural,clip2scene",
                   help="逗号分隔，按 source_id 过滤要抽的论文")
    p.add_argument("--per-paper", type=int, default=3, help="每篇论文取几块")
    args = p.parse_args()
    cfg = load_config(args.config)
    kws = [k.strip() for k in args.keywords.split(",") if k.strip()]
    print(f"抽取范围: {kws}  每篇 {args.per_paper} 块")
    r = build_sample(cfg, kws, args.per_paper)
    print(f"\n共 {r['num_triples']} 三元组, {r['num_entities']} 实体, 来自 {r['num_chunks']} 块")
    print("样例三元组：")
    for t in r["triples"][:20]:
        print(f"  ({t['head']}) --{t['relation']}--> ({t['tail']})")


if __name__ == "__main__":
    main()
