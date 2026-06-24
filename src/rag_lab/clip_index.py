"""
================================================================================
clip_index.py —— 实验：CLIP 图向量检索（架构B） vs 描述法（架构A）
--------------------------------------------------------------------------------
对比两种"让图片可检索"的路线，用的是【同一批已渲染的配图】：

  架构A 描述法（现有主线）：图 → M3 写描述 → 用文本模型(MiniLM)embedding 描述
  架构B CLIP 法（本实验）  ：图 → CLIP 视觉编码器直接 embedding 图片本身
                            查询 → CLIP 多语言文本编码器 → 同一空间里比对

两条路各建一个小索引（图只有几百张，numpy 暴力余弦即可，无需向量库），
在同一组"找某篇论文配图"的问题上比 hit@k —— 用数字看哪种召回更准。

模型（首次会自动下载）：
  图像塔：clip-ViT-B-32
  文本塔：sentence-transformers/clip-ViT-B-32-multilingual-v1（含中文，与图同空间）
================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rag_lab.config import load_config

CLIP_EMB = Path("storage/clip_figures.npy")          # 架构B：CLIP 图向量
CAP_EMB = Path("storage/caption_figures.npy")        # 架构A：MiniLM 描述向量
FIG_META = Path("storage/figures_meta.json")         # 两者共用的图元信息

_MODELS: dict = {}


def _st(name: str):
    """加载并缓存一个 sentence-transformers 模型。"""
    from sentence_transformers import SentenceTransformer
    if name not in _MODELS:
        _MODELS[name] = SentenceTransformer(name)
    return _MODELS[name]


def _collect_figures(cfg: dict) -> list[dict]:
    """从 docs 的 chunk 缓存里，收集所有"配图"块对应的去重图片 + 元信息。"""
    chunks_path = cfg["paths"]["chunks_cache"]
    seen: dict[str, dict] = {}
    for line in open(chunks_path, encoding="utf-8"):
        m = json.loads(line)["metadata"]
        if m.get("modality") == "figure":
            p = m.get("image_path")
            if p and p not in seen and Path(p).exists():
                seen[p] = {
                    "image_path": p,
                    "source_id": m.get("source_id"),
                    "title": m.get("title", ""),
                    "file_name": m.get("file_name", ""),
                    "page": m.get("page"),
                    # 描述文本：从 chunk 文本里拿（架构A 要 embedding 它）
                    "caption": "",
                }
    # 再扫一遍把每张图的描述文本补上（取该图第一个 chunk 的文本）
    for line in open(chunks_path, encoding="utf-8"):
        obj = json.loads(line)
        m = obj["metadata"]
        if m.get("modality") == "figure":
            p = m.get("image_path")
            if p in seen and not seen[p]["caption"]:
                seen[p]["caption"] = obj["text"]
    return list(seen.values())


def build_indexes(cfg: dict) -> int:
    """建两个索引：CLIP 图向量(B) + MiniLM 描述向量(A)。返回图片数。"""
    figs = _collect_figures(cfg)
    if not figs:
        raise RuntimeError("没有配图，先用 configs/docs.yaml ingest（caption=true）")
    paths = [f["image_path"] for f in figs]
    captions = [f["caption"] for f in figs]

    # 架构B：CLIP 编码图片本身
    from PIL import Image
    print(f"[clip] 编码 {len(paths)} 张图片 ...")
    images = [Image.open(p).convert("RGB") for p in paths]
    clip_img = _st("clip-ViT-B-32")
    clip_emb = clip_img.encode(images, normalize_embeddings=True, show_progress_bar=False)

    # 架构A：MiniLM 编码"描述文本"
    print(f"[caption] 编码 {len(captions)} 条描述 ...")
    mini = _st(str(cfg["embedding"]["model"]))
    cap_emb = mini.encode(captions, normalize_embeddings=True, show_progress_bar=False)

    CLIP_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(CLIP_EMB, clip_emb.astype("float32"))
    np.save(CAP_EMB, cap_emb.astype("float32"))
    FIG_META.write_text(json.dumps(figs, ensure_ascii=False), encoding="utf-8")
    print(f"[done] 图片数={len(figs)}  CLIP维度={clip_emb.shape[1]}  描述维度={cap_emb.shape[1]}")
    return len(figs)


def _load():
    meta = json.loads(FIG_META.read_text(encoding="utf-8"))
    return np.load(CLIP_EMB), np.load(CAP_EMB), meta


def _topk(emb_matrix: np.ndarray, query_vec: np.ndarray, k: int) -> list[int]:
    sims = emb_matrix @ query_vec
    return list(sims.argsort()[::-1][:k]), sims


def search_clip(cfg: dict, query: str, k: int = 5):
    """架构B：用 CLIP 多语言文本塔编码查询，搜 CLIP 图向量。"""
    clip_emb, _, meta = _load()
    q = _st("sentence-transformers/clip-ViT-B-32-multilingual-v1").encode(
        [query], normalize_embeddings=True)[0]
    idx, sims = _topk(clip_emb, q, k)
    return [(float(sims[i]), meta[i]) for i in idx]


def search_caption(cfg: dict, query: str, k: int = 5):
    """架构A：用 MiniLM 编码查询，搜描述文本向量。"""
    _, cap_emb, meta = _load()
    q = _st(str(cfg["embedding"]["model"])).encode([query], normalize_embeddings=True)[0]
    idx, sims = _topk(cap_emb, q, k)
    return [(float(sims[i]), meta[i]) for i in idx]


# --- 对比评测：一组"找某篇论文配图"的问题 ----------------------------------
# expected 用论文文件名的关键片段；命中=top-k 里有该论文的图
EVAL = [
    ("图神经网络的四种架构示意图（卷积、循环、自编码、时空）", "Graph Neural networks"),
    ("skeleton 骨架动作识别的时空图网络结构图", "skeleton-based Action"),
    ("CLIP 多模态提示学习 MaPLe 的方法框架图", "MaPLe"),
    ("把 CLIP 用到 3D 点云场景分割的框架", "CLIP2Scene"),
    ("视频文本自适应 CLIP 的多模态提示结构", "Vita-CLIP"),
    ("语音医疗助手的系统框架图", "SpeechMedAssist"),
]


def compare(cfg: dict, k: int = 5) -> None:
    hitA = hitB = 0
    print(f"{'query':36} | A描述法 | B-CLIP")
    print("-" * 64)
    for q, expect in EVAL:
        a = search_caption(cfg, q, k)
        b = search_clip(cfg, q, k)
        a_ok = any(expect.lower() in m["file_name"].lower() for _, m in a)
        b_ok = any(expect.lower() in m["file_name"].lower() for _, m in b)
        hitA += a_ok; hitB += b_ok
        print(f"{q[:36]:36} |   {'✓' if a_ok else '✗'}    |   {'✓' if b_ok else '✗'}")
    n = len(EVAL)
    print("-" * 64)
    print(f"hit@{k}：  架构A(描述法) {hitA}/{n} = {hitA/n:.2f}    架构B(CLIP) {hitB}/{n} = {hitB/n:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="CLIP image retrieval vs caption retrieval.")
    p.add_argument("--config", default="configs/docs.yaml")
    p.add_argument("--build", action="store_true", help="建两个图索引（CLIP + 描述）")
    p.add_argument("--query", default="", help="查一个问题，并排显示两种方法的命中")
    p.add_argument("--compare", action="store_true", help="在内置评测集上对比 hit@k")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()
    cfg = load_config(args.config)

    if args.build:
        build_indexes(cfg)
    if args.query:
        print("=== 架构A 描述法 ===")
        for s, m in search_caption(cfg, args.query, args.k):
            print(f"  {s:.3f}  {m['file_name'][:46]} p.{m['page']}")
        print("=== 架构B CLIP ===")
        for s, m in search_clip(cfg, args.query, args.k):
            print(f"  {s:.3f}  {m['file_name'][:46]} p.{m['page']}")
    if args.compare:
        compare(cfg, args.k)


if __name__ == "__main__":
    main()
