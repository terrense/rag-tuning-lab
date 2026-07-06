"""
================================================================================
generate.py —— L1：基于检索结果生成“带引用”的答案
--------------------------------------------------------------------------------
LLM 用 MiniMax M3（OpenAI 兼容的 chat-completions 接口）。全部配置/环境驱动，
方便改 endpoint 和模型名：
    MINIMAX_API_KEY   （必填，放 .env，永不提交）
    MINIMAX_BASE_URL  （默认 https://api.minimaxi.com/v1）
    MINIMAX_MODEL     （默认 MiniMax-M3）

两条设计原则：
  - 严格“接地”(grounded)：system 提示禁止用资料之外的知识，资料不足要明说。
  - 引用：把检索到的每个块编号 [1..k]，要求模型在结论后标注用到的编号，
    这样答案可溯源到具体资料。
================================================================================
"""

from __future__ import annotations

from typing import Any

from rag_lab.models import SearchHit

# 系统提示词：约束模型“只用资料、可溯源、医疗免责”
SYSTEM_PROMPT = (
    "你是一个严谨的中文医学知识助手。只能依据【资料】中的内容回答问题，"
    "不得使用资料之外的知识，也不得编造资料中没有的信息。"
    "如果资料不足以回答，请直接说明“根据现有资料无法确定”。"
    "回答要简洁、专业；在每个关键结论后用方括号标注引用的资料编号，例如 [1]、[2]。"
    "注意：这些资料来自疾病百科，仅供参考，不能替代医生诊断。"
)


def build_context(hits: list[SearchHit], max_chars: int = 600) -> tuple[str, list[dict]]:
    """把检索到的若干块编号拼成“资料区块”文本，同时返回引用来源清单。"""
    blocks: list[str] = []
    sources: list[dict] = []
    for i, hit in enumerate(hits, start=1):
        title = hit.title or hit.source_id
        text = (hit.text or "")[:max_chars]            # 每块最多取 max_chars，控制 prompt 长度
        blocks.append(f"[{i}] {title}\n{text}")        # 编号 [i] 供模型引用
        sources.append({"n": i, "source_id": hit.source_id, "title": title})
    return "\n\n".join(blocks), sources


def call_minimax(
    cfg: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    role: str = "generate",
) -> dict[str, Any]:
    """历史入口，现在是 rag_lab.llm.chat 的薄壳（名字保留是为了兼容旧调用点）。

    role 决定实际走哪个模型（见 llm.py 的角色路由表）；默认 generate → minimax，
    与旧行为完全一致。graph_*/multimodal/query_rewrite 各自传自己的 role，
    这样 yaml 里 llm.roles.* 一改，全链路的模型分工就换了——A/B 实验的开关。
    """
    from rag_lab.llm import chat

    return chat(cfg, messages, role=role, max_tokens=max_tokens, temperature=temperature)


def _figure_images(hits: list[SearchHit], limit: int) -> list[str]:
    """从命中里挑出“配图”块对应的真实图片路径（去重、限量、确认文件存在）。"""
    import os
    paths: list[str] = []
    for hit in hits:
        if hit.metadata.get("modality") == "figure":
            p = hit.metadata.get("image_path")
            if p and p not in paths and os.path.exists(p):
                paths.append(p)
        if len(paths) >= limit:
            break
    return paths


def generate_answer(cfg: dict[str, Any], query: str, hits: list[SearchHit]) -> dict[str, Any]:
    """给定问题 + 检索到的块，调 MiniMax M3 生成带引用的答案。

    若命中里有“配图”块，则把真实图片一并喂给 M3（图文联合回答 / 真·多模态）。
    """
    gen_cfg = cfg.get("generation", {})
    # 拼 user 提示：问题 + 资料区块
    context, sources = build_context(hits, max_chars=int(gen_cfg.get("context_chars", 600)))
    user_prompt = (
        f"问题：{query}\n\n"
        f"【资料】\n{context}\n\n"
        "请只依据以上资料（含图片）回答，并在关键结论后标注引用编号。"
    )

    # 收集命中的配图图片（可开关、限量）
    images: list[str] = []
    if bool(gen_cfg.get("use_figure_images", True)):
        images = _figure_images(hits, int(gen_cfg.get("max_figure_images", 3)))

    if images:
        # 多模态消息：文字 + 若干图片
        import base64
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        for p in images:
            data = base64.b64encode(open(p, "rb").read()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        try:
            out = call_minimax(cfg, messages)
        except Exception:
            # 图片让请求失败（如尺寸不被接受）就退回纯文字
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}]
            out = call_minimax(cfg, messages)
            images = []
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        out = call_minimax(cfg, messages)

    return {"answer": out["text"], "sources": sources, "model": out["model"],
            "raw_usage": out["usage"], "images_used": images}
