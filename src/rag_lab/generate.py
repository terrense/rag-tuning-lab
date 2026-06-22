"""L1: answer generation with citations, grounded on retrieved chunks.

The LLM is MiniMax M3 via its OpenAI-compatible chat-completions API. Everything
is config/env driven so the endpoint and model id are easy to adjust:

    MINIMAX_API_KEY   (required, kept in .env — never committed)
    MINIMAX_BASE_URL  (default https://api.minimaxi.com/v1)
    MINIMAX_MODEL     (default MiniMax-M3)

Design choices:
- Strictly grounded: the system prompt forbids using outside knowledge and asks
  the model to say so when the context is insufficient.
- Citations: each retrieved chunk is numbered [1..k]; the model must cite the
  numbers it used, so answers stay traceable to sources.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from rag_lab.models import SearchHit

# MiniMax M3 is a reasoning model: it emits a <think>...</think> block before the
# answer. Strip it so callers see only the final answer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    cleaned = _THINK_RE.sub("", text)
    # If the block was truncated (open <think> with no close), drop the prefix.
    if "<think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].replace("<think>", "")
    return cleaned.strip()

SYSTEM_PROMPT = (
    "你是一个严谨的中文医学知识助手。只能依据【资料】中的内容回答问题，"
    "不得使用资料之外的知识，也不得编造资料中没有的信息。"
    "如果资料不足以回答，请直接说明“根据现有资料无法确定”。"
    "回答要简洁、专业；在每个关键结论后用方括号标注引用的资料编号，例如 [1]、[2]。"
    "注意：这些资料来自疾病百科，仅供参考，不能替代医生诊断。"
)


def _load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); does not overwrite existing env."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def build_context(hits: list[SearchHit], max_chars: int = 600) -> tuple[str, list[dict]]:
    """Number the retrieved chunks into a context block + a sources list."""
    blocks: list[str] = []
    sources: list[dict] = []
    for i, hit in enumerate(hits, start=1):
        title = hit.title or hit.source_id
        text = (hit.text or "")[:max_chars]
        blocks.append(f"[{i}] {title}\n{text}")
        sources.append({"n": i, "source_id": hit.source_id, "title": title})
    return "\n\n".join(blocks), sources


def generate_answer(cfg: dict[str, Any], query: str, hits: list[SearchHit]) -> dict[str, Any]:
    _load_dotenv()
    gen_cfg = cfg.get("generation", {})
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1").rstrip("/")
    model = gen_cfg.get("model") or os.environ.get("MINIMAX_MODEL", "MiniMax-M3")
    if not api_key:
        raise RuntimeError(
            "MINIMAX_API_KEY not set. Put it in .env (gitignored) as MINIMAX_API_KEY=..."
        )

    context, sources = build_context(hits, max_chars=int(gen_cfg.get("context_chars", 600)))
    user_prompt = (
        f"问题：{query}\n\n"
        f"【资料】\n{context}\n\n"
        "请只依据以上资料回答，并在关键结论后标注引用编号。"
    )

    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("generation needs 'requests'. Run: pip install requests") from exc

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(gen_cfg.get("temperature", 0.2)),
        # Budget covers the <think> reasoning block plus the final answer.
        "max_tokens": int(gen_cfg.get("max_tokens", 2048)),
    }
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=float(gen_cfg.get("timeout", 60)),
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    answer = _strip_think(raw)
    return {"answer": answer, "sources": sources, "model": model, "raw_usage": data.get("usage")}
