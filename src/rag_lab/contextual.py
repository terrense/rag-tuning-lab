"""
================================================================================
contextual.py —— Contextual Retrieval（Anthropic 2024 的分块增强技法）
--------------------------------------------------------------------------------
普通分块的通病：一个块脱离原文就"失去上下文"。比如一段"治疗以药物和手术为主"，
不知道说的是哪个病；一段表格数字，不知道是哪张表。检索时这种孤块很难被命中。

Anthropic Contextual Retrieval 的做法：**建库时让 LLM 为每个块生成一句"它在
整篇文档里讲的是什么"的上下文前缀，拼到块前面再 embedding**。检索时这句上下文
让孤块重新可被定位。原文报告称能显著降检索失败率。

本项目里：
  · 每个块 → LLM 看"文档标题/摘要 + 该块" → 生成一句上下文 → 前缀拼接
  · 用便宜的 flash（llm.roles.contextual，默认 flash）；带落盘缓存（断点续跑、不重复计费）
  · 有全局封顶 max_chunks——27468 块全做是 2.7 万次调用，示范/A-B 时先跑子集

诚实前提：本项目的疾病语料已经用 prepend_title 把病名塞进每个块了，所以
contextual 的边际收益可能有限——正因如此才要用 A/B 量出来"到底还值不值"，
而不是想当然堆技法。对"块会丢上下文"的语料（如多论文 PDF）它的收益才明显。

用法（建库时）：
  configs 里 chunking.contextual=true，或建库脚本调 augment_chunks()。
================================================================================
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag_lab.llm import chat
from rag_lab.models import Chunk

CACHE_PATH = Path("storage/contextual_cache.json")

_SYS = (
    "你是文档索引助手。给你一篇文档的标题与摘要，以及其中的一个片段。"
    "请用一句不超过30字的中文，说明这个片段在该文档中讲的是什么（补足它脱离原文后"
    "缺失的上下文，如所属主题/对象）。只输出这句话，不要引号、不要解释。"
)


def _doc_brief(doc: dict[str, Any], max_chars: int = 300) -> str:
    title = str(doc.get("title", doc.get("id", "")))
    content = str(doc.get("content", ""))
    return f"标题：{title}\n摘要：{content[:max_chars]}"


def _key(doc_id: str, chunk_text: str) -> str:
    h = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}:{h}"


def augment_chunks(cfg: dict[str, Any], chunks: list[Chunk], docs: list[dict[str, Any]],
                   max_chunks: int = 0) -> dict[str, int]:
    """给 chunks 就地加上下文前缀（改 chunk.text）。返回统计。

    max_chunks>0 时只处理前 N 个（控成本，A/B 或示范用）。带落盘缓存。
    """
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    doc_by_id = {str(d["id"]): d for d in docs}
    n_done = 0
    n_cached = 0
    n_llm = 0
    targets = chunks if max_chunks <= 0 else chunks[:max_chunks]
    for c in targets:
        sid = c.metadata.get("source_id", "")
        doc = doc_by_id.get(sid)
        if not doc:
            continue
        ck = _key(sid, c.text)
        ctx = cache.get(ck)
        if ctx is None:
            out = chat(cfg, [{"role": "system", "content": _SYS},
                             {"role": "user", "content": _doc_brief(doc) + f"\n\n片段：\n{c.text[:400]}"}],
                       role="contextual", max_tokens=80, temperature=0.0)
            ctx = out["text"].strip().replace("\n", " ")
            cache[ck] = ctx
            n_llm += 1
            if n_llm % 20 == 0:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        else:
            n_cached += 1
        if ctx:
            c.text = f"{ctx}\n{c.text}"       # 上下文前缀拼到块前面
            n_done += 1
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return {"augmented": n_done, "llm_calls": n_llm, "cache_hits": n_cached}
