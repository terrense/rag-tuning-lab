"""
================================================================================
graph_build.py —— GraphRAG 第2步：把散三元组拼成知识图 + 实体消歧
--------------------------------------------------------------------------------
第1步抽出的三元组里，同一个实体常有多种写法（CLIP / clip-ViT-B-32 /
Contrastive Language-Image Pre-training；GNN / 图神经网络 / Graph Neural
Networks）。直接建图会变成一堆"孤岛"。这一步：
  1) 实体消歧：把同义写法归并成一个规范节点（normalize + 别名表）
  2) 用 networkx 建有向图（点=实体，边=关系）
  3) 打印图结构：枢纽实体(连接最多)、跨论文桥接实体、某实体的邻居
  4) 存成 graph.json 供第3步检索用
================================================================================
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from rag_lab.config import load_config

TRIPLES_FILE = Path("storage/graph_triples.json")
GRAPH_FILE = Path("storage/graph.json")

# 已知别名：把各种写法的"归一化键"映射到统一显示名
_ALIASES_RAW = {
    "CLIP": ["clip", "clip-vit-b-32", "contrastive language-image pre-training",
             "contrastive language image pretraining"],
    "GNN": ["gnn", "graph neural networks", "graph neural network", "图神经网络",
            "graphneuralnetworks"],
    "Transformer": ["transformer", "transformers"],
}


def _norm_key(s: str) -> str:
    """归一化键：小写 + 去空格/连字符/冒号，用于判断"是不是同一个实体"。"""
    return re.sub(r"[\s\-_:：,.()（）]+", "", s.strip().lower())


# 构造 归一化键 -> 规范显示名 的别名查找表
_ALIAS_LOOKUP = {}
for canon, variants in _ALIASES_RAW.items():
    _ALIAS_LOOKUP[_norm_key(canon)] = canon
    for v in variants:
        _ALIAS_LOOKUP[_norm_key(v)] = canon


def _build_canonical_map(triples: list[dict]) -> dict[str, str]:
    """把所有出现过的实体写法，映射到一个规范显示名（消歧）。"""
    # 收集每个归一化键下出现过的原始写法及频次
    surfaces: dict[str, Counter] = defaultdict(Counter)
    for t in triples:
        for ent in (t["head"], t["tail"]):
            surfaces[_norm_key(ent)][ent] += 1
    canon_map: dict[str, str] = {}
    for key, counter in surfaces.items():
        if key in _ALIAS_LOOKUP:
            display = _ALIAS_LOOKUP[key]          # 别名表优先
        else:
            display = counter.most_common(1)[0][0]  # 否则用最常见的写法当规范名
        for surface in counter:
            canon_map[surface] = display
    return canon_map


def _paper_of(source_id: str) -> str:
    """从 chunk 的 source_id 取出所属论文（去掉 _pNNNN 后缀）。"""
    return re.sub(r"_p\d+.*$", "", source_id or "")


def _embedding_merge(names: list[str], cfg: dict, threshold: float = 0.9) -> dict[str, str]:
    """用 embedding 余弦相似度把语义近义的实体名归并（字符串消歧抓不住的那种）。

    返回 名字 -> 簇代表名（取簇内最短的，通常最干净）。
    """
    import numpy as np
    from rag_lab.embeddings import get_embedder
    names = list(names)
    if len(names) < 2:
        return {n: n for n in names}
    vecs = np.asarray(get_embedder(cfg).embed(names), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True); norms[norms == 0] = 1
    vecs = vecs / norms
    sims = vecs @ vecs.T
    parent = list(range(len(names)))            # 并查集
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sims[i, j] >= threshold:
                parent[find(j)] = find(i)
    clusters: dict[int, list[str]] = {}
    for i, n in enumerate(names):
        clusters.setdefault(find(i), []).append(n)
    mapping = {}
    for members in clusters.values():
        rep = min(members, key=len)             # 簇代表 = 最短名
        for m in members:
            mapping[m] = rep
    return mapping


def build_graph(triples: list[dict], cfg: dict | None = None,
                embed_dedup: bool = False, threshold: float = 0.9) -> nx.MultiDiGraph:
    canon = _build_canonical_map(triples)
    if embed_dedup and cfg is not None:
        # 在字符串消歧之上，再用 embedding 合并语义近义的规范名
        merge = _embedding_merge(sorted(set(canon.values())), cfg, threshold)
        canon = {surface: merge.get(disp, disp) for surface, disp in canon.items()}
    g = nx.MultiDiGraph()
    for t in triples:
        h = canon[t["head"]]
        ta = canon[t["tail"]]
        paper = _paper_of(t.get("source_id", ""))
        for node in (h, ta):
            if not g.has_node(node):
                g.add_node(node, papers=set())
            g.nodes[node]["papers"].add(paper)
        g.add_edge(h, ta, relation=t["relation"], paper=paper)
    return g


def summarize(g: nx.MultiDiGraph) -> None:
    print(f"\n图规模：{g.number_of_nodes()} 个实体节点, {g.number_of_edges()} 条关系边")

    # 枢纽实体：连接（入度+出度）最多的
    deg = sorted(g.degree(), key=lambda x: x[1], reverse=True)
    print("\n枢纽实体（连接最多 Top10）：")
    for node, d in deg[:10]:
        print(f"  {d:3d} 连接  {node}")

    # 跨论文桥接实体：在多于一篇论文里出现的实体（GraphRAG 多跳的关键）
    bridges = [(n, len(g.nodes[n]["papers"])) for n in g.nodes
               if len(g.nodes[n]["papers"]) > 1]
    bridges.sort(key=lambda x: x[1], reverse=True)
    print(f"\n跨论文桥接实体（出现在 >1 篇论文，共 {len(bridges)} 个）：")
    for n, c in bridges[:10]:
        print(f"  {c} 篇  {n}")

    # 看一个枢纽实体的邻居（演示"图上能走到哪"）
    hub = "CLIP" if g.has_node("CLIP") else (deg[0][0] if deg else None)
    if hub:
        print(f"\n实体「{hub}」的关系邻居：")
        seen = set()
        for _, tail, data in g.out_edges(hub, data=True):
            line = f"  ({hub}) --{data['relation']}--> ({tail})"
            if line not in seen:
                print(line); seen.add(line)
        for head, _, data in g.in_edges(hub, data=True):
            line = f"  ({head}) --{data['relation']}--> ({hub})"
            if line not in seen:
                print(line); seen.add(line)


def save_graph(g: nx.MultiDiGraph) -> None:
    """存成 node-link JSON（把 set 转 list 以便序列化），供第3步检索。"""
    h = g.copy()
    for n in h.nodes:
        h.nodes[n]["papers"] = sorted(h.nodes[n]["papers"])
    data = nx.node_link_data(h, edges="links")
    GRAPH_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n图已存到 {GRAPH_FILE}")


def main() -> None:
    p = argparse.ArgumentParser(description="GraphRAG step 2: build + dedup graph.")
    p.add_argument("--triples", default=str(TRIPLES_FILE))
    p.add_argument("--config", default="configs/docs.yaml")
    p.add_argument("--embed-dedup", action="store_true", help="额外用 embedding 合并语义近义实体")
    p.add_argument("--threshold", type=float, default=0.9)
    args = p.parse_args()
    data = json.loads(Path(args.triples).read_text(encoding="utf-8"))
    triples = data["triples"]
    print(f"读入 {len(triples)} 个三元组" + ("（含 embedding 消歧）" if args.embed_dedup else ""))
    cfg = load_config(args.config) if args.embed_dedup else None
    g = build_graph(triples, cfg=cfg, embed_dedup=args.embed_dedup, threshold=args.threshold)
    summarize(g)
    save_graph(g)


if __name__ == "__main__":
    main()
