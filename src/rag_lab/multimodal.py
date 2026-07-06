"""
================================================================================
multimodal.py —— 多模态 PDF 导入（路线图 L2~L5）
--------------------------------------------------------------------------------
把一篇 PDF 拆成三种“可检索单元”，统一变成文档 dict，复用现有文本向量库：
  · 文字：逐页抽取（fitz）
  · 表格：pdfplumber 抽取 → Markdown 表
  · 配图：渲染“图多的页”为整图 → MiniMax M3 视觉生成描述 → 描述进检索库

设计要点 / 取舍：
  - PDF 常把一张图拆成成百上千个小图元，按图元抽取会爆炸；所以按“页渲染”理解。
  - 图片 captioning 用 M3，每张图一次 API：所以做了 全局封顶 + 描述落盘缓存
    （再次 ingest 不会重复调用），并优先选“图像对象最多的页”。
  - 一切语言化成文本进同一个向量库；figure 块的 metadata 里留有图片路径，
    回答时可据此把真实图片喂回 M3 做图文联合回答。
================================================================================
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

FIGURE_DIR = Path("storage/figures")
CAPTION_CACHE = Path("storage/figure_captions.json")

_CAPTION_SYS = (
    "你是文档图表理解助手。用简洁中文描述这张图片表达的核心信息："
    "图表类型（折线/柱状/流程图/示意图/表格截图等）、坐标轴或字段、"
    "主要趋势或结论、涉及的关键实体或术语。只输出描述本身，不要客套。"
)


def _slug(path: Path) -> str:
    s = re.sub(r"[^a-z0-9一-鿿]+", "_", path.stem.lower()).strip("_")
    return (s or "doc")[:60]


# --- 描述缓存（落盘，避免重复调用 M3）---------------------------------------
def _load_caption_cache() -> dict[str, str]:
    if CAPTION_CACHE.exists():
        try:
            return json.loads(CAPTION_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_caption_cache(cache: dict[str, str]) -> None:
    CAPTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CAPTION_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# --- 表格抽取 ----------------------------------------------------------------
def _tables_to_markdown(path: Path, min_rows: int) -> dict[int, list[str]]:
    """用 pdfplumber 抽每页的表格，转成 Markdown。返回 {页码: [markdown表, ...]}。"""
    out: dict[int, list[str]] = {}
    try:
        import pdfplumber
    except ImportError:
        return out
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                mds = []
                for tbl in tables:
                    rows = [[(c or "").strip().replace("\n", " ") for c in row] for row in tbl if row]
                    if len(rows) < min_rows:
                        continue
                    # 过滤掉空表/伪表：非空单元格太少（pdfplumber 常把无边框排版误判成表）
                    non_empty = sum(1 for r in rows for c in r if c)
                    if non_empty < 4 or len(rows[0]) < 2:
                        continue
                    header = rows[0]
                    md = "| " + " | ".join(header) + " |\n"
                    md += "| " + " | ".join("---" for _ in header) + " |\n"
                    for r in rows[1:]:
                        # 补齐/截断到表头列数
                        r = (r + [""] * len(header))[: len(header)]
                        md += "| " + " | ".join(r) + " |\n"
                    mds.append(md)
                if mds:
                    out[i] = mds
    except Exception:
        pass
    return out


# --- 图片：渲染整页 + M3 描述 ------------------------------------------------
def _render_page_png(page, out_path: Path, zoom: float, max_px: int) -> bool:
    """把一页渲染成 PNG 存盘。返回是否成功。"""
    try:
        import fitz  # noqa
        pix = page.get_pixmap(matrix=__import__("fitz").Matrix(zoom, zoom))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_path))
        # 长边超过 max_px 就缩一下，控制 base64 体积
        try:
            from PIL import Image
            im = Image.open(out_path)
            if max(im.size) > max_px:
                ratio = max_px / max(im.size)
                im = im.resize((int(im.size[0] * ratio), int(im.size[1] * ratio)))
                im.save(out_path)
        except Exception:
            pass
        return True
    except Exception:
        return False


def _caption_image(cfg: dict, png_path: Path) -> str:
    """调 MiniMax M3 视觉，给一张图生成中文描述。"""
    from rag_lab.generate import call_minimax
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    data_uri = f"data:image/png;base64,{data}"
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": _CAPTION_SYS},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]},
    ]
    out = call_minimax(cfg, messages, max_tokens=400, role="caption")
    return out["text"].strip()


def load_pdf_multimodal(
    path: Path, root: Path, cfg: dict[str, Any], budget: dict[str, int], caption_cache: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """把一篇 PDF 拆成 文字/表格/配图 三类文档。budget 跨文档共享，控制 figure 总量。"""
    import fitz

    mm = cfg.get("multimodal", {})
    do_tables = bool(mm.get("tables", True))
    do_figures = bool(mm.get("figures", True))
    do_caption = bool(mm.get("caption", True))
    per_doc_cap = int(mm.get("max_figures_per_doc", 8))
    zoom = float(mm.get("render_zoom", 2.0))
    max_px = int(mm.get("max_image_px", 1400))
    min_rows = int(mm.get("min_table_rows", 2))

    rel = path.relative_to(root).as_posix() if _is_relative(path, root) else path.name
    slug = _slug(path)
    docs: list[dict[str, Any]] = []
    counts = {"pdf_text_pages": 0, "pdf_tables": 0, "pdf_figures": 0}

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        print(f"  [skip] 打不开 {path.name}: {exc}")
        return docs, counts

    # 1) 文字：逐页
    fig_candidates: list[tuple[int, int]] = []   # (页码, 图像对象数)
    for i in range(doc.page_count):
        page = doc[i]
        text = (page.get_text() or "").strip()
        if len(text) >= 30:
            docs.append({
                "id": f"pdf_{slug}_p{i+1:04d}",
                "title": f"{path.stem} p.{i+1}",
                "tags": ["doc", "pdf", "text"],
                "content": text,
                "metadata": {"source_type": "pdf_text", "file_name": path.name,
                             "path": rel, "page": i + 1, "modality": "text"},
            })
            counts["pdf_text_pages"] += 1
        n_imgs = len(page.get_images())
        if n_imgs > 0:
            fig_candidates.append((i, n_imgs))

    # 2) 表格
    if do_tables:
        for page_no, mds in _tables_to_markdown(path, min_rows).items():
            for j, md in enumerate(mds):
                docs.append({
                    "id": f"pdf_{slug}_p{page_no:04d}_tbl{j}",
                    "title": f"{path.stem} 表格 p.{page_no}",
                    "tags": ["doc", "pdf", "table"],
                    "content": f"{path.stem} 第{page_no}页表格：\n{md}",
                    "metadata": {"source_type": "pdf_table", "file_name": path.name,
                                 "path": rel, "page": page_no, "modality": "table"},
                })
                counts["pdf_tables"] += 1

    # 3) 配图：选“图最多的页”，渲染整页 + M3 描述（受全局/单文档上限约束）
    if do_figures:
        fig_candidates.sort(key=lambda x: x[1], reverse=True)   # 图多的页优先
        used = 0
        for page_idx, _n in fig_candidates:
            if used >= per_doc_cap or budget["remaining"] <= 0:
                break
            png = FIGURE_DIR / slug / f"p{page_idx+1:04d}.png"
            if not png.exists() and not _render_page_png(doc[page_idx], png, zoom, max_px):
                continue
            key = png.as_posix()
            caption = caption_cache.get(key)               # 先查缓存
            if caption is None and do_caption:
                try:
                    caption = _caption_image(cfg, png)
                    caption_cache[key] = caption           # 写缓存
                except Exception as exc:
                    print(f"  [caption fail] {png.name}: {exc}")
                    caption = ""
            caption = caption or f"{path.stem} 第{page_idx+1}页的图（暂无描述）"
            docs.append({
                "id": f"pdf_{slug}_p{page_idx+1:04d}_fig",
                "title": f"{path.stem} 配图 p.{page_idx+1}",
                "tags": ["doc", "pdf", "figure"],
                "content": f"{path.stem} 第{page_idx+1}页配图：{caption}",
                "metadata": {"source_type": "pdf_figure", "file_name": path.name,
                             "path": rel, "page": page_idx + 1, "modality": "figure",
                             "image_path": key},
            })
            counts["pdf_figures"] += 1
            used += 1
            budget["remaining"] -= 1

    doc.close()
    return docs, counts


def _is_relative(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_multimodal_corpus(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """遍历 docs_dir 下所有 PDF，做多模态导入。figure 总量受 max_figures_total 约束。"""
    docs_dir = Path(cfg["paths"].get("docs_dir", "data/docs"))
    pdfs = sorted(p for p in docs_dir.rglob("*.pdf") if p.is_file())
    mm = cfg.get("multimodal", {})
    budget = {"remaining": int(mm.get("max_figures_total", 60))}
    caption_cache = _load_caption_cache()

    all_docs: list[dict[str, Any]] = []
    totals = {"pdf_files": 0, "pdf_text_pages": 0, "pdf_tables": 0, "pdf_figures": 0}
    for pdf in pdfs:
        print(f"  [pdf] {pdf.name} ...")
        docs, counts = load_pdf_multimodal(pdf, docs_dir, cfg, budget, caption_cache)
        all_docs.extend(docs)
        totals["pdf_files"] += 1
        for k in ("pdf_text_pages", "pdf_tables", "pdf_figures"):
            totals[k] += counts.get(k, 0)
        _save_caption_cache(caption_cache)   # 每篇都存一次，断点可续

    return all_docs, totals
