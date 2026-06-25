"""
================================================================================
graph_query.py —— GraphRAG 第3步：用知识图回答问题
--------------------------------------------------------------------------------
把第2步建好的图真正用起来，回答"关系型/多跳"问题：
  1) 实体定位：从问题里认出图中的实体（如"CLIP"）
  2) 子图遍历：抓这些实体的邻居关系（1~2 跳），形成"关系链"
  3) 生成：把关系链作为证据喂给 MiniMax，让它据此回答并说清关系

和普通 RAG 的区别：普通 RAG 给 LLM 的是"相似文本块"，这里给的是
"结构化的实体关系"——所以能回答"A 和 B 怎么联系起来""谁基于谁"这类问题。
================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from rag_lab.config import load_config
from rag_lab.generate import call_minimax
from rag_lab.graph_build import GRAPH_FILE, _ALIAS_LOOKUP, _norm_key


def load_graph() -> nx.MultiDiGraph:
    data = json.loads(Path(GRAPH_FILE).read_text(encoding="utf-8"))
    return nx.node_link_graph(data, multigraph=True, directed=True, edges="links")


def link_entities(query: str, g: nx.MultiDiGraph) -> list[str]:
    """从问题里认出图中实体：归一化子串匹配 + 别名表。"""
    qkey = _norm_key(query)
    matched = []
    for node in g.nodes:
        k = _norm_key(node)
        if len(k) >= 2 and k in qkey:          # 节点名（归一化后）出现在问题里
            matched.append(node)
    # 别名：问题里出现 "图神经网络" 这种变体，映射到规范节点
    for variant_key, canon in _ALIAS_LOOKUP.items():
        if variant_key in qkey and g.has_node(canon) and canon not in matched:
            matched.append(canon)
    # 去重，优先保留更长（更具体）的实体
    matched = sorted(set(matched), key=lambda n: len(n), reverse=True)
    return matched


def gather_facts(g: nx.MultiDiGraph, entities: list[str], hops: int = 1) -> list[dict]:
    """从命中实体出发，收集 hops 跳内的关系边（去重）。"""
    facts = []
    seen = set()
    frontier = set(entities)
    visited = set()
    for _ in range(hops):
        nxt = set()
        for node in frontier:
            if node in visited or not g.has_node(node):
                continue
            visited.add(node)
            for _, tail, d in g.out_edges(node, data=True):
                key = (node, d["relation"], tail)
                if key not in seen:
                    seen.add(key); facts.append({"head": node, "relation": d["relation"],
                                                 "tail": tail, "paper": d.get("paper", "")})
                nxt.add(tail)
            for head, _, d in g.in_edges(node, data=True):
                key = (head, d["relation"], node)
                if key not in seen:
                    seen.add(key); facts.append({"head": head, "relation": d["relation"],
                                                 "tail": node, "paper": d.get("paper", "")})
                nxt.add(head)
        frontier = nxt - visited
    return facts


def answer(cfg: dict, query: str, hops: int = 1) -> dict:
    g = load_graph()
    entities = link_entities(query, g)
    facts = gather_facts(g, entities, hops=hops)
    if not facts:
        return {"entities": entities, "facts": [], "answer": "知识图中没找到相关实体/关系。"}

    fact_lines = "\n".join(f"({f['head']}) --{f['relation']}--> ({f['tail']})" for f in facts)
    papers = sorted({f["paper"] for f in facts if f["paper"]})
    sys = ("你是基于知识图谱的问答助手。只能依据给定的【关系事实】回答，"
           "要说清实体之间的关系链；不要编造关系。资料不足就直说。")
    user = f"问题：{query}\n\n【关系事实】\n{fact_lines}\n\n请据此回答。"
    out = call_minimax(cfg, [{"role": "system", "content": sys},
                             {"role": "user", "content": user}], max_tokens=1024)
    return {"entities": entities, "facts": facts, "papers": papers, "answer": out["text"]}


def main() -> None:
    p = argparse.ArgumentParser(description="GraphRAG step 3: answer via the graph.")
    p.add_argument("--config", default="configs/docs.yaml")
    p.add_argument("--query", required=True)
    p.add_argument("--hops", type=int, default=1)
    args = p.parse_args()
    cfg = load_config(args.config)
    r = answer(cfg, args.query, hops=args.hops)
    print(f"命中实体：{r['entities']}")
    print(f"\n抓到 {len(r['facts'])} 条关系事实（{args.hops}跳）：")
    for f in r["facts"][:20]:
        print(f"  ({f['head']}) --{f['relation']}--> ({f['tail']})")
    print(f"\n回答：\n{r['answer']}")


if __name__ == "__main__":
    main()
