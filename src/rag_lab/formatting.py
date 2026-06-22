"""
================================================================================
formatting.py —— 把结果美化成人类可读的终端输出
--------------------------------------------------------------------------------
纯展示层，不含检索逻辑。供 ingest / query 等 CLI 打印用。
================================================================================
"""

from __future__ import annotations

from typing import Any

from rag_lab.models import SearchHit


def snippet(text: str, limit: int) -> str:
    """把多行文本压成一行，超过 limit 就截断加省略号。"""
    compact = " ".join(text.split())                 # 折叠所有空白/换行成单空格
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def print_ingest_summary(summary: dict[str, Any]) -> None:
    """打印建库统计（ingest 完成后）。"""
    print("Ingest complete")
    print(f"  store       : {summary['store_type']}")
    print(f"  collection  : {summary['collection']}")
    print(f"  docs/chunks : {summary['docs']} docs / {summary['chunks']} chunks")
    source_counts = summary.get("source_counts", {})
    if source_counts:
        print(
            "  sources     : "
            f"cards={source_counts.get('interview_cards', 0)} "
            f"files={source_counts.get('files', 0)} "
            f"pdf_pages={source_counts.get('pdf_pages', 0)}"
        )
    print(f"  dimension   : {summary['dimension']}")   # 向量维度
    print(f"  store count : {summary['store_count']}")  # 库里实际存了多少条
    print(f"  chunks cache: {summary['chunks_cache']}")


def print_query_result(result: dict[str, Any], snippet_chars: int = 220) -> None:
    """打印一次查询的结果（问题 + 本次配置 + top_k 命中）。"""
    print("Query")
    print(f"  {result['query']}")
    cfg = result["config"]
    print(
        "Config\n"
        f"  store={cfg['store']} collection={cfg['collection']} "
        f"chunk={cfg['chunk_size']}/{cfg['chunk_overlap']} "
        f"candidate_k={cfg['candidate_k']} top_k={cfg['top_k']} "
        f"hybrid={cfg['hybrid']} rerank={cfg['rerank']}"
    )
    print("Results")
    for idx, hit in enumerate(result["hits"], start=1):
        print(_format_hit(idx, hit, snippet_chars))


def _format_hit(idx: int, hit: SearchHit, snippet_chars: int) -> str:
    """格式化单条命中：各路分数 + 来源 + 排名细节 + 正文摘要。"""
    # 某路没参与就显示 '-'
    vector = "-" if hit.vector_score is None else f"{hit.vector_score:.4f}"
    bm25 = "-" if hit.bm25_score is None else f"{hit.bm25_score:.4f}"
    rerank = "-" if hit.rerank_score is None else f"{hit.rerank_score:.4f}"
    ranks = ",".join(f"{k}={v}" for k, v in sorted(hit.rank_details.items()))
    return (
        f"\n[{idx}] score={hit.score:.4f} vector={vector} bm25={bm25} rerank={rerank}\n"
        f"    source={hit.source_id} title={hit.title}\n"
        f"    file={_format_file_ref(hit.metadata)}\n"
        f"    ranks={ranks or '-'}\n"
        f"    {snippet(hit.text, snippet_chars)}"
    )


def _format_file_ref(metadata: dict[str, Any]) -> str:
    """如果命中来自文件，显示“文件路径 + PDF 页码”；否则显示 '-'。"""
    path = metadata.get("path")
    if not path:
        return "-"
    page = metadata.get("page")
    if page:
        return f"{path} p.{page}"
    return str(path)
