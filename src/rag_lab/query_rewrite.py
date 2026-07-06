"""
================================================================================
query_rewrite.py —— 三层级联 query 改写（在检索之前重写问题）
--------------------------------------------------------------------------------
为什么需要：原样检索会被三类问题拖偏——
  · 语义鸿沟：用户说“拉肚子”，资料写“腹泻”；用户口语，资料是术语。
  · 意图模糊：一种问法抓不全，需要多角度/补全。
  · 历史指代：多轮里“它/这个病”指代上文，必须先消解成具体实体。

三层级联（从便宜到贵，每层可独立开关，按顺序施加）：
  第1层 规则      ：同义词/口语→术语 补全（确定性、零依赖、最快）
  第2层 传统NLP   ：分词 + 去停用词，整理出“关键词查询”（更利于 BM25）
  第3层 大模型    ：MiniMax M3 改写——rewrite(消解指代/规范意图) / hyde / multi

输出：一组 (向量检索用文本, BM25检索用文本) 的 spec 列表，交给 pipeline 去检索。
  · 向量端偏好自然语言；BM25 端偏好关键词——所以第2层只整理 BM25 那一侧。
================================================================================
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# 第1层：规则改写 —— 口语/同义词 → 标准术语（命中就把术语“补”到查询里，不删原词）
# 这张表可以持续扩充；也可以以后改成从数据文件加载。
# ---------------------------------------------------------------------------
SYNONYMS: dict[str, str] = {
    "拉肚子": "腹泻", "跑肚": "腹泻", "拉稀": "腹泻",
    "发烧": "发热", "低烧": "低热", "高烧": "高热",
    "心慌": "心悸", "心脏不舒服": "心悸 胸闷",
    "喘不上气": "呼吸困难 气促", "气短": "呼吸困难", "上不来气": "呼吸困难",
    "肚子疼": "腹痛", "肚子痛": "腹痛", "胃疼": "胃痛 上腹痛",
    "头疼": "头痛", "嗓子疼": "咽痛", "没劲": "乏力", "浑身没劲": "乏力",
    "瘦了": "消瘦 体重下降", "眼睛黄": "黄疸", "皮肤黄": "黄疸",
    "起夜": "夜尿增多", "尿多": "多尿",
    "脚趾疼": "关节痛", "关节疼": "关节痛",
    "长水疱": "疱疹", "起红疹": "皮疹", "起疹子": "皮疹",
    "怕冷": "畏寒", "怕热": "怕热 多汗",
}


def layer1_rules(query: str) -> str:
    """规则层：把查询里出现的口语词对应的标准术语补到末尾（扩展，不替换）。"""
    extras: list[str] = []
    for colloquial, term in SYNONYMS.items():
        if colloquial in query:
            for t in term.split():               # 术语可能有多个（如 心悸 胸闷）
                if t not in query and t not in extras:
                    extras.append(t)
    return query if not extras else f"{query} {' '.join(extras)}"


# ---------------------------------------------------------------------------
# 第2层：传统NLP —— 分词 + 去停用词，整理成关键词查询（主要给 BM25 用）
# 有 jieba 用 jieba 分词；没有就降级用一个简单的清洗（去问句词/标点）。
# ---------------------------------------------------------------------------
STOPWORDS: set[str] = {
    "的", "了", "吗", "呢", "啊", "是", "在", "和", "与", "或", "我", "你", "他", "它",
    "会", "不", "会不会", "是不是", "有没有", "怎么", "怎样", "如何", "哪些", "哪个",
    "什么", "为什么", "需要", "应该", "可能", "一般", "通常", "请问", "谢谢", "吧",
    "这个", "那个", "一下", "可以", "能", "要", "做", "看", "挂", "该",
}
# 降级清洗时要剥掉的问句词/标点
_PUNCT_RE = re.compile(r"[，。？！、；：,.?!;:\s]+")


def layer2_nlp(text: str) -> str:
    """传统NLP层：分词去停用词，返回空格连接的关键词串。"""
    try:
        import jieba  # 可选依赖：装了才用，分词质量更好
        tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        keywords = [t for t in tokens if t not in STOPWORDS and not _PUNCT_RE.fullmatch(t)]
        return " ".join(keywords) if keywords else text
    except Exception:
        # 降级：没有 jieba 时，去标点 + 删掉整词命中的停用词（粗略但有效）
        cleaned = _PUNCT_RE.sub(" ", text)
        for sw in sorted(STOPWORDS, key=len, reverse=True):   # 长词先删，避免子串误伤
            cleaned = cleaned.replace(sw, " ")
        result = " ".join(w for w in cleaned.split() if w)
        return result or text


# ---------------------------------------------------------------------------
# 第3层：大模型改写（MiniMax M3）—— 解决意图模糊、语义鸿沟、历史指代
# ---------------------------------------------------------------------------
_REWRITE_SYS = (
    "你是医疗检索的查询改写器。把用户的问题改写成一个适合检索的、规范的中文查询："
    "用医学标准术语替换口语；消除歧义、明确意图；如果给了对话历史，"
    "把其中的指代（如“它”“这个病”“上面说的”）替换成具体的疾病/实体名。"
    "只输出改写后的查询本身，不要任何解释、引号或前缀。"
)
_MULTI_SYS = (
    "你是医疗检索助手。针对用户问题，生成 {n} 个语义等价但表述不同的检索查询"
    "（同义词、口语与专业两种说法、不同侧重点）。每行一个，不要编号、不要解释。"
)
_HYDE_SYS = (
    "你是医学专家。针对问题写一段简短的、像疾病百科条目的“假设答案”段落，"
    "尽量具体（可包含可能的疾病名、典型症状、检查项目），即使不完全确定也照写。"
    "只输出这段文字，不要解释。"
)


def _history_block(history: list[str] | None) -> str:
    """把对话历史拼成提示片段（用于消解指代）。"""
    if not history:
        return ""
    return "对话历史：\n" + "\n".join(history) + "\n\n"


def layer3_rewrite(cfg: dict[str, Any], query: str, history: list[str] | None) -> str:
    """大模型把问题改写成一条规范查询（消解历史指代、统一术语）。"""
    from rag_lab.generate import call_minimax
    user = f"{_history_block(history)}当前问题：{query}"
    out = call_minimax(
        cfg,
        [{"role": "system", "content": _REWRITE_SYS}, {"role": "user", "content": user}],
        max_tokens=256,
        role="rewrite",
    )
    return out["text"].strip() or query


def layer3_multi(cfg: dict[str, Any], query: str, history: list[str] | None, n: int) -> list[str]:
    """大模型生成 n 个不同表述的查询（多查询召回）。"""
    from rag_lab.generate import call_minimax
    user = f"{_history_block(history)}当前问题：{query}"
    out = call_minimax(
        cfg,
        [{"role": "system", "content": _MULTI_SYS.format(n=n)}, {"role": "user", "content": user}],
        max_tokens=256,
        role="rewrite",
    )
    variants = [line.strip(" -·•\t") for line in out["text"].splitlines() if line.strip()]
    return variants[:n]


def layer3_hyde(cfg: dict[str, Any], query: str, history: list[str] | None) -> str:
    """大模型写一段“假设答案”，拿它去做向量检索（HyDE，补向量召回短板）。"""
    from rag_lab.generate import call_minimax
    user = f"{_history_block(history)}问题：{query}"
    out = call_minimax(
        cfg,
        [{"role": "system", "content": _HYDE_SYS}, {"role": "user", "content": user}],
        max_tokens=int(cfg.get("query", {}).get("hyde_max_tokens", 256)),
        role="rewrite",
    )
    return out["text"].strip() or query


# ---------------------------------------------------------------------------
# 总入口：按配置施加三层，产出检索用的 (向量文本, BM25文本) spec 列表
# ---------------------------------------------------------------------------
# 缓存：同一个 (配置, 查询, 历史) 不重复调 LLM（实验里会重复跑同一题）
_CACHE: dict[tuple, list[tuple[str, str]]] = {}


def build_retrieval_queries(
    cfg: dict[str, Any], query: str, history: list[str] | None = None
) -> list[tuple[str, str]]:
    """返回若干 (vector_text, bm25_text)。不开任何层时就是 [(原问题, 原问题)]。"""
    qcfg = cfg.get("query", {})
    use_rules = bool(qcfg.get("rules", False))
    use_nlp = bool(qcfg.get("nlp", False))
    llm = str(qcfg.get("llm", "none")).lower()

    if not use_rules and not use_nlp and llm == "none":
        return [(query, query)]                         # 完全不改写，保持原行为

    cache_key = (use_rules, use_nlp, llm, int(qcfg.get("num_variants", 3)),
                 query, tuple(history or ()))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # 第1层：规则补全（对向量、BM25 都有益）
    base = layer1_rules(query) if use_rules else query

    # 第3层：大模型（rewrite 在这里就把 base 改写掉；hyde/multi 在下面产生变体）
    if llm == "rewrite":
        base = layer3_rewrite(cfg, base, history)

    if llm == "hyde":
        vector_texts = [layer3_hyde(cfg, base, history)]   # 向量端用“假设答案”
        bm25_base = base                                   # BM25 端仍用真实问题/术语
    elif llm == "multi":
        vector_texts = [base] + layer3_multi(cfg, base, history, int(qcfg.get("num_variants", 3)))
        bm25_base = None                                   # 每个变体自己做 BM25
    else:
        vector_texts = [base]
        bm25_base = base

    # 第2层：传统NLP 只整理 BM25 那一侧（向量端保留自然语言）
    specs: list[tuple[str, str]] = []
    for vt in vector_texts:
        bt = bm25_base if bm25_base is not None else vt    # hyde 用固定 bm25_base；multi 用各自
        bt = layer2_nlp(bt) if use_nlp else bt
        specs.append((vt, bt))

    _CACHE[cache_key] = specs
    return specs
