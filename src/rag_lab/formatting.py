from __future__ import annotations

from typing import Any

from rag_lab.models import SearchHit


def snippet(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def print_ingest_summary(summary: dict[str, Any]) -> None:
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
    print(f"  dimension   : {summary['dimension']}")
    print(f"  store count : {summary['store_count']}")
    print(f"  chunks cache: {summary['chunks_cache']}")


def print_query_result(result: dict[str, Any], snippet_chars: int = 220) -> None:
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
    path = metadata.get("path")
    if not path:
        return "-"
    page = metadata.get("page")
    if page:
        return f"{path} p.{page}"
    return str(path)
