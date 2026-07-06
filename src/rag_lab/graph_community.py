"""
================================================================================
graph_community.py —— GraphRAG 第4步：社区检测 + 摘要（回答全局问题）
--------------------------------------------------------------------------------
有些问题没有"局部答案"，比如"我这批论文整体在研究哪几个方向？"——任何单个
文本块或单条关系都答不了，它要的是"全局概览"。

GraphRAG 的做法：
  1) 社区检测：把知识图聚成若干"簇"（社区）——联系紧密的实体抱团
  2) 社区摘要：让 MiniMax 给每个社区写一句主题概括
  3) 全局回答：把所有社区摘要汇总起来，回答全局问题

注意：我们的图较稀疏（每篇论文只抽了几块、跨论文连接少），所以社区基本=按论文
聚团——这是抽取密度的局限，抽得越多社区越能体现"主题"而非"单篇"。
================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from rag_lab.config import load_config
from rag_lab.generate import call_minimax
from rag_lab.graph_build import GRAPH_FILE
from rag_lab.graph_query import load_graph

COMM_FILE = Path("storage/communities.json")


def detect_communities(g: nx.MultiDiGraph) -> list[list[str]]:
    """把有向多重图投影成无向简单图，用贪心模块度做社区检测。"""
    ug = nx.Graph()
    ug.add_nodes_from(g.nodes())
    for h, t in g.edges():
        if h != t:                                   # 忽略自环
            ug.add_edge(h, t)
    comms = greedy_modularity_communities(ug)
    # 按大小降序，过滤掉只有 1 个点的孤立社区
    return [sorted(c) for c in sorted(comms, key=len, reverse=True) if len(c) >= 2]


def _community_facts(g: nx.MultiDiGraph, members: set) -> list[str]:
    """收集社区内部（首尾都在社区里）的关系，作为摘要素材。"""
    facts = []
    for h, t, d in g.edges(data=True):
        if h in members and t in members and h != t:
            facts.append(f"({h}) --{d['relation']}--> ({t})")
    return facts


def summarize_communities(cfg: dict) -> dict:
    """检测社区 + 给每个社区写主题摘要，存盘。"""
    g = load_graph()
    comms = detect_communities(g)
    out = []
    for i, members in enumerate(comms, 1):
        mset = set(members)
        facts = _community_facts(g, mset)
        sys = ("你是知识图谱社区分析器。根据给定实体和它们之间的关系，"
               "用一句话概括这个社区的主题（研究方向/领域），再给一个简短名称。"
               '只输出 JSON：{"name":"", "summary":""}')
        user = "实体：" + "、".join(members[:25]) + "\n关系：\n" + "\n".join(facts[:30])
        try:
            res = call_minimax(cfg, [{"role": "system", "content": sys},
                                     {"role": "user", "content": user}], max_tokens=300, role="graph")
            txt = res["text"]
            s, e = txt.find("{"), txt.rfind("}")
            meta = json.loads(txt[s:e+1]) if s >= 0 else {}
        except Exception:
            meta = {}
        out.append({"id": i, "size": len(members), "members": members,
                    "name": meta.get("name", f"社区{i}"),
                    "summary": meta.get("summary", "（摘要失败）")})
        print(f"  社区{i} ({len(members)}个实体): {meta.get('name','?')} —— {meta.get('summary','')[:60]}")
    COMM_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"communities": out, "num": len(out)}


def global_answer(cfg: dict, query: str) -> str:
    """用所有社区摘要回答全局问题（map-reduce 里的 reduce 步）。"""
    if not COMM_FILE.exists():
        summarize_communities(cfg)
    comms = json.loads(COMM_FILE.read_text(encoding="utf-8"))
    blocks = "\n".join(f"[社区{c['id']}] {c['name']}：{c['summary']}（含 {c['size']} 个实体）"
                       for c in comms)
    sys = ("你是基于知识图谱社区摘要的全局问答助手。"
           "依据下面各社区的主题摘要，综合回答用户的全局性问题。")
    user = f"问题：{query}\n\n【各社区主题摘要】\n{blocks}\n\n请综合回答。"
    return call_minimax(cfg, [{"role": "system", "content": sys},
                              {"role": "user", "content": user}], max_tokens=1024, role="graph")["text"]


def main() -> None:
    p = argparse.ArgumentParser(description="GraphRAG step 4: communities + global QA.")
    p.add_argument("--config", default="configs/docs.yaml")
    p.add_argument("--summarize", action="store_true", help="检测社区并写摘要")
    p.add_argument("--query", default="", help="回答一个全局性问题")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.summarize:
        r = summarize_communities(cfg)
        print(f"\n共 {r['num']} 个社区，已存到 {COMM_FILE}")
    if args.query:
        print(f"\n全局回答：\n{global_answer(cfg, args.query)}")


if __name__ == "__main__":
    main()
