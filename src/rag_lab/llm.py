"""
================================================================================
llm.py —— 统一的多模型 LLM 客户端（OpenAI 兼容 chat-completions）
--------------------------------------------------------------------------------
项目里现在有三个可用模型，按"任务难度 × 成本"分工：

  别名             环境变量                      分工
  minimax          MINIMAX_*                    多模态（唯一能吃图的）+ 对标 pro 的生成
  deepseek-pro     DEEPSEEK_MODEL_PRO           难任务：LLM-as-judge、生成对比、质检过滤
  deepseek-flash   DEEPSEEK_MODEL_FLASH         便宜跑量：query 改写、批量出题、打分初筛

为什么要"角色路由"而不是到处写死模型名：
  - A/B 实验的需要：比如 E5 要测「query 改写用 flash 够不够」，只要在 yaml 里写
    llm.roles.rewrite=deepseek-flash 就能切，代码一行不动，实验才可复现。
  - 裁判独立原则：LLM-as-judge 的裁判必须 ≠ 被评的生成模型（自偏袒），
    路由表让"谁生成、谁评判"成为显式配置而非隐式约定。

用法：
    from rag_lab.llm import chat
    out = chat(cfg, messages, role="rewrite")            # 按角色走路由表
    out = chat(cfg, messages, model="deepseek-pro")      # 或直接点名
    # out = {"text": 已剥思考的正文, "usage": token统计, "model": 实际模型id, "alias": 别名}
================================================================================
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

# 推理模型（MiniMax M3 / DeepSeek 思考模式）会在答案前输出 <think>...</think>，剥掉。
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# 角色 → 默认别名。yaml 里 llm.roles.<角色>=<别名> 可逐个覆盖。
# generate/caption/rewrite/graph 默认 minimax 是为了不改变已有 baseline 的行为；
# judge/filter 默认 pro、evalgen 默认 flash 是新能力，按分工原则定。
DEFAULT_ROLES: dict[str, str] = {
    "generate": "minimax",    # 带引用的答案生成（可能带图 → 必须能多模态）
    "caption": "minimax",     # 配图描述（必须能多模态）
    "rewrite": "minimax",     # query 改写（E5 实验会用 flash 来挑战它）
    "graph": "minimax",       # 图三元组抽取 / 社区摘要
    "judge": "deepseek-pro",  # LLM-as-judge（裁判 ≠ 生成者）
    "filter": "deepseek-pro", # 评测集质检过滤
    "evalgen": "deepseek-flash",  # 批量生成评测问题（量大、任务简单）
}


def load_dotenv(path: str | Path = ".env") -> None:
    """极简 .env 加载器（KEY=VALUE 逐行）。不覆盖已存在的环境变量。"""
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


def strip_think(text: str) -> str:
    """去掉 <think>...</think>；思考块被截断时丢弃其前缀。"""
    cleaned = _THINK_RE.sub("", text)
    if "<think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].replace("<think>", "")
    return cleaned.strip()


def _resolve(alias: str, cfg: dict[str, Any] | None) -> dict[str, str]:
    """别名 → {api_key, base_url, model}。全部从环境变量取（.env 已加载）。"""
    if alias == "minimax":
        # generation.model 保留旧的覆盖入口（早于 llm.py 的配置方式）
        gen_model = (cfg or {}).get("generation", {}).get("model")
        return {
            "api_key": os.environ.get("MINIMAX_API_KEY", ""),
            "base_url": os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
            "model": gen_model or os.environ.get("MINIMAX_MODEL", "MiniMax-M3"),
            "key_name": "MINIMAX_API_KEY",
        }
    if alias in ("deepseek-pro", "deepseek-flash"):
        env = "DEEPSEEK_MODEL_PRO" if alias == "deepseek-pro" else "DEEPSEEK_MODEL_FLASH"
        default = "deepseek-v4-pro" if alias == "deepseek-pro" else "deepseek-v4-flash"
        return {
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "model": os.environ.get(env, default),
            "key_name": "DEEPSEEK_API_KEY",
        }
    raise ValueError(f"Unknown llm alias '{alias}'. Known: minimax / deepseek-pro / deepseek-flash")


def resolve_role(cfg: dict[str, Any] | None, role: str) -> str:
    """角色 → 别名：先查 yaml 的 llm.roles，再落到 DEFAULT_ROLES。"""
    roles = ((cfg or {}).get("llm", {}) or {}).get("roles", {}) or {}
    return str(roles.get(role) or DEFAULT_ROLES.get(role, "minimax"))


def chat(
    cfg: dict[str, Any] | None,
    messages: list[dict[str, Any]],
    *,
    role: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    """统一入口：按 role 路由（或 model 点名别名），带重试，返回已剥思考的正文。

    重试策略：429/5xx/网络错误按 2s·4s 退避重试；4xx（参数错）不重试直接抛。
    """
    load_dotenv()
    alias = model or resolve_role(cfg, role or "generate")
    prov = _resolve(alias, cfg)
    if not prov["api_key"]:
        raise RuntimeError(f"{prov['key_name']} not set. Put it in .env (gitignored).")

    import requests

    gen_cfg = (cfg or {}).get("generation", {})
    payload = {
        "model": prov["model"],
        "messages": messages,
        "temperature": float(gen_cfg.get("temperature", 0.2)) if temperature is None else temperature,
        # max_tokens 要装下 <think> + 正文，否则截断
        "max_tokens": int(gen_cfg.get("max_tokens", 2048)) if max_tokens is None else max_tokens,
    }
    timeout = float(gen_cfg.get("timeout", 90)) if timeout is None else timeout

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{prov['base_url'].rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {prov['api_key']}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 429 or resp.status_code >= 500:  # 限流/服务端错 → 可重试
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()                                  # 其它 4xx → 直接抛（不重试）
            data = resp.json()
            raw = data["choices"][0]["message"]["content"] or ""
            return {"text": strip_think(raw), "usage": data.get("usage"),
                    "model": prov["model"], "alias": alias}
        except requests.HTTPError as exc:
            m = re.search(r"\b(\d{3})\b", str(exc))
            status = int(m.group(1)) if m else 0
            retriable = status == 429 or status >= 500
            if not retriable or attempt >= retries:
                raise
            last_err = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= retries:
                raise
            last_err = exc
        time.sleep(2 * (attempt + 1))          # 2s, 4s 退避
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_err}")
