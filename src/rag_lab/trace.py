"""
================================================================================
trace.py —— 轻量可观测性：分阶段延迟 + token 用量 + 成本估算
--------------------------------------------------------------------------------
生产 RAG 上线后，"哪一步慢、哪一步烧钱"必须能看见。这个模块提供一个 Trace 对象：

  - span(name)  ：上下文管理器，记一个阶段耗时（embed/retrieve/rerank/generate...）
  - add_usage() ：累加某模型的 token 用量（从 LLM 响应的 usage 抠）
  - cost()      ：按价目表把 token 折算成钱（估算，价目可配）
  - to_dict()   ：整条 trace 序列化，随响应返回 + 落 logs/traces.jsonl

设计成"无侵入"：不开 trace 时代码照常跑（llm.chat 里对 None trace 是 no-op）。
一条 trace 串起一次请求的所有阶段，这就是分布式追踪(Langfuse/OTel)的最小内核。
================================================================================
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 价目表：USD / 1M tokens，(输入价, 输出价)。**估算值，按需在这里改**。
# 三个模型分工不同，成本对比正是"轻任务用便宜模型"的量化依据。
PRICING: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash": (0.10, 0.30),
    "deepseek-v4-pro": (0.55, 2.19),
    "MiniMax-M3": (0.30, 1.20),
}
_DEFAULT_PRICE = (0.50, 1.50)   # 未知模型的兜底价


@dataclass
class Span:
    name: str
    ms: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """一次请求的追踪记录。线程安全（并发请求各持一个）。"""
    request_id: str = ""
    query: str = ""
    spans: list[Span] = field(default_factory=list)
    usage: dict[str, dict[str, int]] = field(default_factory=dict)  # model -> {prompt, completion}
    meta: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @contextmanager
    def span(self, name: str, **meta: Any):
        """计时一个阶段：with trace.span('retrieve'): ..."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self.spans.append(Span(name=name, ms=ms, meta=meta))

    def add_usage(self, model: str, usage: dict[str, Any] | None) -> None:
        """累加某模型的 token 用量（OpenAI 兼容 usage: prompt_tokens/completion_tokens）。"""
        if not usage:
            return
        with self._lock:
            slot = self.usage.setdefault(model, {"prompt": 0, "completion": 0})
            slot["prompt"] += int(usage.get("prompt_tokens", 0) or 0)
            slot["completion"] += int(usage.get("completion_tokens", 0) or 0)

    def cost_usd(self) -> float:
        """按价目表折算总成本（估算）。"""
        total = 0.0
        for model, u in self.usage.items():
            pin, pout = PRICING.get(model, _DEFAULT_PRICE)
            total += u["prompt"] / 1e6 * pin + u["completion"] / 1e6 * pout
        return total

    def total_ms(self) -> float:
        return sum(s.ms for s in self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "total_ms": round(self.total_ms(), 1),
            "spans": [{"name": s.name, "ms": round(s.ms, 1), **s.meta} for s in self.spans],
            "usage": self.usage,
            "cost_usd": round(self.cost_usd(), 6),
            **self.meta,
        }

    def append_jsonl(self, path: str | Path = "logs/traces.jsonl") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
