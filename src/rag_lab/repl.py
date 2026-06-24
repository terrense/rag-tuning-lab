"""
================================================================================
repl.py —— 常驻问答（热启动）：模型只加载一次，反复提问
--------------------------------------------------------------------------------
痛点：CLI 每次 ask/query 都是新进程，要重新加载 embedding + cross-encoder 两个
      模型（~30s），所以每问一次都等 40s。
办法：这个 REPL 进程不退出，启动时把模型/索引都加载好（warmup），之后每次提问
      复用缓存——第二问起只剩“检索 ~2s + LLM 生成”。

用法：
  python -m rag_lab.repl --config configs/docs.yaml            # 检索+生成
  python -m rag_lab.repl --config configs/diseases.yaml --no-llm  # 只检索，最快
  输入问题回车即可；输入 exit / quit / 空行 退出。
================================================================================
"""

from __future__ import annotations

import argparse
import time

from rag_lab.config import get_path, load_config, parse_value, set_dotted
from rag_lab.embeddings import get_embedder
from rag_lab.pipeline import _get_retrieval_assets, query_config
from rag_lab.generate import generate_answer


def warmup(cfg: dict) -> None:
    """启动时把该加载的都加载好：embedding 模型、chunk+BM25 索引、cross-encoder。"""
    t0 = time.perf_counter()
    get_embedder(cfg)                                   # 加载 embedding 模型
    _get_retrieval_assets(get_path(cfg, "chunks_cache"))  # 读 chunk + 建 BM25
    # 跑一次假查询，把 cross-encoder 也加载好（这样第一个真问题就快）
    try:
        query_config(cfg, "warmup")
    except Exception:
        pass
    print(f"[warmup] 模型与索引就绪，用时 {time.perf_counter() - t0:.1f}s\n")


def answer_one(cfg: dict, question: str, no_llm: bool) -> None:
    """回答一个问题，并打印检索/生成各自耗时。"""
    t0 = time.perf_counter()
    result = query_config(cfg, question)
    t_ret = time.perf_counter() - t0

    if no_llm:
        print(f"\n检索结果（{t_ret*1000:.0f}ms）：")
        for i, h in enumerate(result["hits"], 1):
            m = h.metadata
            tag = m.get("modality") or m.get("source_type", "")
            print(f"  [{i}] {tag:7} {h.title}  score={h.score:.3f}")
        return

    t1 = time.perf_counter()
    gen = generate_answer(cfg, question, result["hits"])
    t_gen = time.perf_counter() - t1

    print(f"\n回答（检索 {t_ret:.1f}s + 生成 {t_gen:.1f}s）：")
    print(gen["answer"])
    print("\n引用来源：")
    for s in gen["sources"]:
        print(f"  [{s['n']}] {s['title']}  ({s['source_id']})")
    if gen.get("images_used"):
        print("看图：", ", ".join(gen["images_used"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm interactive RAG REPL (load models once).")
    parser.add_argument("--config", default="configs/docs.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--no-llm", action="store_true", help="只检索，不调用 LLM 生成（最快）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw_value))

    print(f"[repl] config={args.config}  no_llm={args.no_llm}")
    warmup(cfg)

    while True:
        try:
            q = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit", "q"}:
            break
        try:
            answer_one(cfg, q, args.no_llm)
        except Exception as exc:
            print(f"[error] {exc}")
        print()
    print("bye.")


if __name__ == "__main__":
    main()
