"""
================================================================================
semantic_cache.py —— 语义缓存：相似问题直接命中，跳过检索+生成
--------------------------------------------------------------------------------
生产 RAG 降本降延迟的头号手段。精确字符串缓存没用（用户很少一字不差重复问），
但"苯中毒有什么症状" 和 "长期接触苯会中毒吗有哪些表现" 语义上是同一问 →
用 query 的向量做余弦匹配，超过阈值就复用上次的答案。

关键取舍：
  - 阈值太低 → 张冠李戴（把不同问题的答案返给用户），比不缓存还糟 → 默认 0.95 偏保守。
  - 只缓存"生成成功且未拒答"的结果（拒答/报错不该被固化）。
  - 复用同一个 embedder（和检索共用），不额外加载模型。
  - 落盘持久化（重启不丢），也支持纯内存。

命中率 / 节省延迟都统计，好在实验里量化"缓存到底省了多少"。
================================================================================
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class SemanticCache:
    def __init__(self, embedder: Any, threshold: float = 0.90,
                 path: str | Path | None = "storage/semantic_cache.jsonl",
                 max_entries: int = 5000):
        self.embedder = embedder
        self.threshold = float(threshold)
        self.path = Path(path) if path else None
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._queries: list[str] = []
        self._vecs: np.ndarray | None = None       # (N, dim) L2-normalized
        self._payloads: list[dict] = []
        self.stats = {"lookups": 0, "hits": 0, "ms_saved": 0.0}
        if self.path and self.path.exists():
            self._load()

    # --- 内部：向量归一化后按行堆叠，余弦=点积 ---
    def _embed(self, text: str) -> np.ndarray:
        v = np.asarray(self.embedder.embed([text])[0], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _load(self) -> None:
        rows = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows[-self.max_entries:]:
            self._queries.append(r["query"])
            self._payloads.append(r["payload"])
        if self._queries:
            self._vecs = np.vstack([self._embed(q) for q in self._queries])

    def lookup(self, query: str) -> dict | None:
        """返回缓存的 payload（命中）或 None。命中时把它标记为 cached。"""
        with self._lock:
            self.stats["lookups"] += 1
            if self._vecs is None or len(self._queries) == 0:
                return None
            qv = self._embed(query)
            sims = self._vecs @ qv                      # 余弦相似度（都已归一化）
            i = int(np.argmax(sims))
            if float(sims[i]) >= self.threshold:
                self.stats["hits"] += 1
                self.stats["ms_saved"] += float(self._payloads[i].get("_gen_ms", 0.0))
                out = dict(self._payloads[i])
                out["_cache"] = {"hit": True, "similarity": round(float(sims[i]), 4),
                                 "matched_query": self._queries[i]}
                return out
            return None

    def put(self, query: str, payload: dict) -> None:
        """把一次成功生成的结果写入缓存（含它的生成耗时，用于统计节省）。"""
        with self._lock:
            qv = self._embed(query)
            self._queries.append(query)
            self._payloads.append(payload)
            self._vecs = qv[None, :] if self._vecs is None else np.vstack([self._vecs, qv])
            # 超上限：丢最旧
            if len(self._queries) > self.max_entries:
                self._queries.pop(0); self._payloads.pop(0)
                self._vecs = self._vecs[1:]
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"query": query, "payload": payload,
                                        "ts": time.time()}, ensure_ascii=False) + "\n")

    def hit_rate(self) -> float:
        return self.stats["hits"] / self.stats["lookups"] if self.stats["lookups"] else 0.0

    def summary(self) -> dict:
        return {"entries": len(self._queries), "threshold": self.threshold,
                "hit_rate": round(self.hit_rate(), 3), **self.stats}
