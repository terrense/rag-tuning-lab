"""
================================================================================
serve.py —— 生产级服务层：把实验台包成能上线的 HTTP 服务
--------------------------------------------------------------------------------
从"CLI 一次性进程"升级到"常驻服务"，这是"生产级"叙事最缺的一块。要点：

  · 模型常驻       启动时预热 embedder/reranker/chunks（lifespan），请求不再冷启动
  · 分阶段可观测   每个请求一条 Trace：embed/retrieve/rerank/generate 各阶段延迟
                   + token 用量 + 成本估算，随响应返回并落 logs/traces.jsonl
  · 语义缓存       相似问题直接命中，跳过检索+生成（命中率/省时可查）
  · 流式输出       /ask/stream 用 SSE 逐字返回（首 token 延迟体验）
  · 健康与指标     /health、/metrics（缓存命中率、累计请求、p50/p95 延迟）

端点：
  GET  /health          存活 + 预热状态
  GET  /metrics         服务级指标（缓存、延迟分位）
  POST /search          只检索，返回命中 + trace（最快，无 LLM）
  POST /ask             检索 + 带引用生成，返回答案 + sources + trace
  POST /ask/stream      同上，SSE 流式

跑：
  $py = "C:/Users/Administrator/miniconda3/envs/rag-tuning-lab/python.exe"
  & $py -m uvicorn rag_lab.serve:app --host 127.0.0.1 --port 8000
  # 或： & $py -m rag_lab.serve  --config configs/diseases.yaml
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_lab.config import load_config
from rag_lab.trace import Trace

# 服务全局状态（lifespan 里填充；单进程内所有请求共享）
STATE: dict[str, Any] = {"cfg": None, "cache": None, "embedder": None,
                         "latencies": deque(maxlen=1000), "requests": 0}

DEFAULT_CONFIG = os.environ.get("RAG_CONFIG", "configs/diseases.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热：加载配置 + embedder + 检索资产 + 语义缓存。请求就不再冷启动。"""
    cfg = load_config(os.environ.get("RAG_CONFIG", DEFAULT_CONFIG))
    STATE["cfg"] = cfg
    from rag_lab.embeddings import get_embedder
    from rag_lab.config import get_path
    from rag_lab.pipeline import _get_retrieval_assets

    embedder = get_embedder(cfg)
    STATE["embedder"] = embedder
    # 预热检索资产（chunks + BM25），第一次请求就不用等
    try:
        _get_retrieval_assets(get_path(cfg, "chunks_cache"))
    except Exception as exc:  # 没 ingest 也能起服务，查询时再报
        print(f"[warn] retrieval assets not preloaded: {exc}")
    # 语义缓存开关（config.cache.enabled，默认开）
    cache_cfg = cfg.get("cache", {})
    if cache_cfg.get("enabled", True):
        from rag_lab.semantic_cache import SemanticCache
        STATE["cache"] = SemanticCache(
            embedder, threshold=float(cache_cfg.get("threshold", 0.90)),
            path=cache_cfg.get("path", "storage/semantic_cache.jsonl"))
    print(f"[ready] config={cfg.get('_config_path')} cache={'on' if STATE['cache'] else 'off'}")
    yield
    print("[shutdown]")


app = FastAPI(title="RAG Tuning Lab API", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str
    history: list[str] | None = None
    top_k: int | None = None
    no_cache: bool = False
    no_llm: bool = False


def _run_retrieval(cfg: dict, query: str, history: list[str] | None, trace: Trace) -> dict:
    """检索阶段（带 trace 计时），返回 query_config 的结果。"""
    from rag_lab.pipeline import query_config
    with trace.span("retrieve"):
        result = query_config(cfg, query, history=history)
    trace.meta["candidate_count"] = result.get("candidate_count")
    return result


def _serialize_hits(hits: list, limit: int) -> list[dict]:
    out = []
    for h in hits[:limit]:
        out.append({"source_id": h.source_id, "title": h.title,
                    "score": round(float(h.score), 4),
                    "text": (h.text or "")[:300]})
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "warm": STATE["embedder"] is not None,
            "config": (STATE["cfg"] or {}).get("_config_path"),
            "cache": STATE["cache"] is not None}


@app.get("/metrics")
def metrics() -> dict:
    lat = sorted(STATE["latencies"])
    def pct(p: float) -> float:
        if not lat:
            return 0.0
        return round(lat[min(len(lat) - 1, int(p / 100 * len(lat)))], 1)
    out = {"requests": STATE["requests"], "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99)}}
    if STATE["cache"]:
        out["cache"] = STATE["cache"].summary()
    return out


@app.post("/search")
def search(req: AskRequest) -> dict:
    """只检索，不生成——最快路径，给"只要命中文档"的场景。"""
    cfg = STATE["cfg"]
    trace = Trace(request_id=str(uuid.uuid4())[:8], query=req.query)
    t0 = time.perf_counter()
    result = _run_retrieval(cfg, req.query, req.history, trace)
    STATE["requests"] += 1
    STATE["latencies"].append((time.perf_counter() - t0) * 1000)
    trace.append_jsonl()
    top_k = req.top_k or int(cfg["retrieval"].get("top_k", 5))
    return {"query": req.query, "hits": _serialize_hits(result["hits"], top_k),
            "trace": trace.to_dict()}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """检索 + 带引用生成。走语义缓存。"""
    cfg = STATE["cfg"]
    cache = None if (req.no_cache or STATE["cache"] is None) else STATE["cache"]
    t0 = time.perf_counter()

    # 1) 语义缓存命中 → 直接返回
    if cache is not None:
        cached = cache.lookup(req.query)
        if cached is not None:
            STATE["requests"] += 1
            STATE["latencies"].append((time.perf_counter() - t0) * 1000)
            return cached

    trace = Trace(request_id=str(uuid.uuid4())[:8], query=req.query)
    result = _run_retrieval(cfg, req.query, req.history, trace)
    top_k = req.top_k or int(cfg["retrieval"].get("top_k", 5))
    hits = result["hits"][:top_k]

    answer_payload: dict[str, Any]
    if req.no_llm:
        answer_payload = {"answer": None, "sources": []}
    else:
        from rag_lab.generate import generate_answer
        gen_t0 = time.perf_counter()
        with trace.span("generate"):
            gen = generate_answer(cfg, req.query, hits, trace=trace)
        gen_ms = (time.perf_counter() - gen_t0) * 1000
        answer_payload = {"answer": gen["answer"], "sources": gen["sources"],
                          "model": gen.get("model"), "_gen_ms": gen_ms}

    STATE["requests"] += 1
    STATE["latencies"].append((time.perf_counter() - t0) * 1000)
    trace.append_jsonl()
    payload = {"query": req.query, **answer_payload,
               "hits": _serialize_hits(hits, top_k), "trace": trace.to_dict(),
               "_cache": {"hit": False}}
    # 只缓存成功生成、非拒答的结果
    if cache is not None and payload.get("answer") and "无法确定" not in payload["answer"]:
        cache.put(req.query, {k: v for k, v in payload.items() if k != "trace"})
    return payload


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """SSE 流式：先发 sources，再逐块发答案，末尾发 trace。体验上首 token 更快。"""
    cfg = STATE["cfg"]
    trace = Trace(request_id=str(uuid.uuid4())[:8], query=req.query)

    def gen():
        result = _run_retrieval(cfg, req.query, req.history, trace)
        top_k = req.top_k or int(cfg["retrieval"].get("top_k", 5))
        hits = result["hits"][:top_k]
        yield _sse("sources", _serialize_hits(hits, top_k))
        from rag_lab.generate import generate_answer_stream
        with trace.span("generate"):
            for piece in generate_answer_stream(cfg, req.query, hits, trace=trace):
                yield _sse("token", piece)
        STATE["requests"] += 1
        STATE["latencies"].append(trace.total_ms())
        trace.append_jsonl()
        yield _sse("trace", trace.to_dict())
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the RAG API server.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    os.environ["RAG_CONFIG"] = args.config
    import uvicorn
    uvicorn.run("rag_lab.serve:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
