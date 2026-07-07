# -*- coding: utf-8 -*-
"""
================================================================================
ocr_table_eval.py —— E13 OCR 臂：扫描件上的表格结构化提取评测
--------------------------------------------------------------------------------
输入是 data/tables/scan/*.jpg（旋转+噪声+模糊+JPEG 的仿扫描件），没有文本层，
表格结构必须从像素恢复——t3(多级表头)/t5(窄列换行) 这两个数字解析免疫的坑，
在这里才真正开考。

引擎（--engine）：
  ppstructure  PaddleOCR PP-StructureV3：版面分析+表格识别，直接输出带
               rowspan/colspan 的 HTML → 展开成网格
  rapidocr     RapidOCR 只给文字框（无表格结构）→ 我们自己做 y/x 聚类重建
               行列（教学用"裸 OCR"基线，预期在 t5 上翻车）

评分与 pdfplumber 臂共用 table_eval.py 的 grids_to_rows + score（口径一致）。

⚠ 用 ocr-lab 环境跑：
  C:/Users/Administrator/miniconda3/envs/ocr-lab/python.exe -X utf8 scripts/ocr_table_eval.py
================================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_eval import TDIR, grids_to_rows, score  # noqa: E402


# ---------------------------------------------------------------------------
# HTML 表格 → 网格（处理 rowspan/colspan：占位矩阵展开，值写满整个跨度——
# 表格识别模型敢输出 rowspan，说明它已经"理解"了合并，值就该归属每一行）
# ---------------------------------------------------------------------------
class _TableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict]] = []
        self._cell: dict | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = {"text": "", "rowspan": int(a.get("rowspan", 1) or 1),
                          "colspan": int(a.get("colspan", 1) or 1)}
            self.rows[-1].append(self._cell)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"] += data


def html_to_grid(html: str) -> list[list[str]]:
    p = _TableHTML()
    p.feed(html)
    if not p.rows:
        return []
    grid: dict[tuple[int, int], str] = {}
    occupied: set[tuple[int, int]] = set()
    for r, cells in enumerate(p.rows):
        c = 0
        for cell in cells:
            while (r, c) in occupied:
                c += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    occupied.add((r + dr, c + dc))
                    grid[(r + dr, c + dc)] = cell["text"].strip()
            c += cell["colspan"]
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]


# ---------------------------------------------------------------------------
# 引擎 1：PP-StructureV3（表格识别 → HTML）
# ---------------------------------------------------------------------------
_PP_PIPE = None


def grids_ppstructure(img: Path) -> list[list[list[str]]]:
    global _PP_PIPE
    from paddleocr import PPStructureV3
    if _PP_PIPE is None:
        # 扫描件有旋转 → 打开方向分类；unwarp 对平扫没必要。
        # enable_mkldnn=False：paddlepaddle 3.x Windows 的 PIR+oneDNN 执行器
        # 会抛 ConvertPirAttribute2RuntimeAttribute NotImplementedError。
        _PP_PIPE = PPStructureV3(use_doc_orientation_classify=True, use_doc_unwarping=False,
                                 device="cpu", enable_mkldnn=False)
    grids = []
    for res in _PP_PIPE.predict(str(img)):
        d = res.json if isinstance(res.json, dict) else {}
        r = d.get("res", d)
        for t in r.get("table_res_list", []):
            html = t.get("pred_html") or ""
            g = html_to_grid(html)
            if g:
                grids.append(g)
    return grids


# ---------------------------------------------------------------------------
# 引擎 2：RapidOCR 文字框 + 自研 y/x 聚类（"裸 OCR"基线）
# ---------------------------------------------------------------------------
_RAPID = None


def grids_rapidocr(img: Path) -> list[list[list[str]]]:
    """boxes → 行：按框中心 y 聚类（阈值=中位框高×0.7）；行内按 x 排序后，
    用全页的 x 中心聚类出列坐标，再把每个框分配到最近的列。
    这正是面试稿里说的"根据 bbox 的 y 轴聚类重建行、x 轴聚类重建列"——
    也正是 t5 要打崩的东西（换行把一逻辑行拆成两条 y 带）。"""
    global _RAPID
    from rapidocr_onnxruntime import RapidOCR
    if _RAPID is None:
        _RAPID = RapidOCR()
    result, _ = _RAPID(str(img))
    if not result:
        return []
    boxes = []
    for pts, text, _conf in result:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        boxes.append({"x": sum(xs) / 4, "y": sum(ys) / 4,
                      "h": max(ys) - min(ys), "text": text.strip()})
    boxes.sort(key=lambda b: b["y"])
    med_h = sorted(b["h"] for b in boxes)[len(boxes) // 2]
    # y 聚类成行
    lines: list[list[dict]] = []
    for b in boxes:
        if lines and abs(b["y"] - lines[-1][0]["y"]) < med_h * 0.7:
            lines[-1].append(b)
        else:
            lines.append([b])
    # 全页 x 聚类出列中心（用行内框的 x，间距 < 40px 归同列）
    xs = sorted(b["x"] for b in boxes)
    col_centers: list[float] = []
    for x in xs:
        if not col_centers or x - col_centers[-1] > 40:
            col_centers.append(x)
        else:
            col_centers[-1] = (col_centers[-1] + x) / 2
    grid = []
    for line in lines:
        row = [""] * len(col_centers)
        for b in sorted(line, key=lambda t: t["x"]):
            ci = min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - b["x"]))
            row[ci] = (row[ci] + " " + b["text"]).strip()
        grid.append(row)
    return [grid]


# ---------------------------------------------------------------------------
# 引擎 3：GOT-OCR2.0（端到端 VLM：图 → 带格式文本，表格输出 LaTeX/HTML）
# ---------------------------------------------------------------------------
_GOT = None


def _clean_latex_math(s: str) -> str:
    """GOT 把单位/数值包在数学模式里：\\(10^{\\sim}9/\\mathrm{L}\\) → 10^9/L。
    逐条还原成 GT 用的普通文本（GT 用 10^9/L、↑↓、μ）。"""
    s = s.replace(r"\(", "").replace(r"\)", "").replace("$", "")
    s = s.replace(r"\downarrow", "↓").replace(r"\uparrow", "↑")
    s = s.replace(r"\sim", "").replace("~", "")            # 10^{\sim}9 里的 ~ 是识别噪声
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\{([^}]*)\}", r"^\1", s)                # 10^{9} → 10^9
    s = re.sub(r"_\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)                      # 其余残留控制序列
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", "", s).strip()


def _split_cells(line: str) -> list[str]:
    """按未转义的 & 分列（表格里的 & 都是列分隔；GT 文本不含 &）。"""
    return [c.strip() for c in line.split("&")]


def _latex_tabular_to_grid(tex: str) -> list[list[str]]:
    r"""\begin{tabular} 块 → 网格。处理 \multicolumn（横向展开）、
    \multirow（纵向填充：值写满该列的 N 行，等价 fill-down）、数学模式清洗、
    以及单元格里嵌套 tabular（t5 的换行会被 GOT 表示成小 tabular，取其文本拼接）。"""
    grids = []
    # 最外层 tabular（用计数配对，避免嵌套 tabular 提前闭合）
    depth, start = 0, None
    blocks = []
    for m in re.finditer(r"\\(begin|end)\{tabular\}", tex):
        if m.group(1) == "begin":
            if depth == 0:
                start = m.end()
            depth += 1
        else:
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(tex[start:m.start()])
    for body in blocks:
        # 去掉列格式说明 {|c|c|...}（紧跟 \begin{tabular} 的那段），行线
        body = re.sub(r"^\s*\{[^{}]*\}", "", body)
        body = re.sub(r"\\[hc]line(\[[^]]*\])?", "", body)
        # 先把嵌套 tabular 压平成其纯文本（消除内部 \\ 和 & 的干扰）
        body = re.sub(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
                      lambda mm: _clean_latex_math(mm.group(1).replace("&", "").replace(r"\\", "")),
                      body, flags=re.DOTALL)
        grid: list[list[str]] = []
        # multirow 续行填充：LaTeX 对被跨的列会留一个空占位 &，所以按"列位置"
        # 记忆值，续行在该列遇到空格时回填（不能自动插值，否则整行右移）。
        active: dict[int, tuple[int, str]] = {}   # 列位置 → (剩余续行数, 值)
        for raw in re.split(r"\\\\", body):
            raw = raw.strip()
            if not raw:
                continue
            row: list[str] = []
            for cell in _split_cells(raw):
                col = len(row)
                mr = re.match(r"\\multirow\{(-?\d+)\}\{[^}]*\}\{(.*)\}$", cell, re.DOTALL)
                mc = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", cell, re.DOTALL)
                if mr:
                    val = _clean_latex_math(mr.group(2))
                    row.append(val)
                    if abs(int(mr.group(1))) > 1:
                        active[col] = (abs(int(mr.group(1))) - 1, val)
                elif mc:
                    row.append(_clean_latex_math(mc.group(2)))
                    row.extend([""] * (int(mc.group(1)) - 1))
                else:
                    val = _clean_latex_math(cell)
                    if not val and col in active:      # 空占位 + 该列有活跃 multirow → 回填
                        n, v = active[col]
                        val = v
                        active[col] = (n - 1, v) if n - 1 > 0 else None  # type: ignore
                        if active[col] is None:
                            del active[col]
                    row.append(val)
            grid.append(row)
        if grid:
            grids.append(grid)
    return grids


def grids_gotocr(img: Path) -> list[list[list[str]]]:
    global _GOT
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if _GOT is None:
        name = "stepfun-ai/GOT-OCR-2.0-hf"
        proc = AutoProcessor.from_pretrained(name)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForImageTextToText.from_pretrained(
            name, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
            device_map=dev)
        _GOT = (proc, model, dev)
    proc, model, dev = _GOT
    from PIL import Image
    image = Image.open(img).convert("RGB")
    import torch
    inputs = proc(image, return_tensors="pt", format=True).to(dev)   # format=True → 结构化输出
    with torch.no_grad():
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=4096,
                             tokenizer=proc.tokenizer, stop_strings="<|im_end|>")
    text = proc.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    (TDIR / "raw").mkdir(exist_ok=True)
    (TDIR / "raw" / f"{img.stem}_got.txt").write_text(text, encoding="utf-8")
    if "<table" in text.lower() or "<tr" in text.lower():
        g = html_to_grid(text)
        return [g] if g else []
    return _latex_tabular_to_grid(text)


# ---------------------------------------------------------------------------
# 引擎 4：DeepSeek-OCR（VLM → Markdown；与 GOT 同为端到端路线，对照第二个 VLM）
# ---------------------------------------------------------------------------
_DSOCR = None


def _md_table_to_grid(md: str) -> list[list[str]]:
    """Markdown 管道表 → 网格。跳过 |---|---| 分隔线。"""
    grid = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):   # 分隔行
            continue
        grid.append(cells)
    return grid


def grids_dsocr(img: Path) -> list[list[list[str]]]:
    global _DSOCR
    import torch
    from transformers import AutoModel, AutoTokenizer
    if _DSOCR is None:
        name = "deepseek-ai/DeepSeek-OCR"
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(name, trust_remote_code=True,
                                          _attn_implementation="eager",
                                          torch_dtype=torch.bfloat16, use_safetensors=True)
        model = model.eval().cuda().to(torch.bfloat16)
        _DSOCR = (tok, model)
    tok, model = _DSOCR
    outdir = TDIR / "raw" / "_dsocr_tmp"
    outdir.mkdir(parents=True, exist_ok=True)
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    text = model.infer(tok, prompt=prompt, image_file=str(img), output_path=str(outdir),
                       base_size=1024, image_size=640, crop_mode=True, save_results=False)
    text = text if isinstance(text, str) else str(text)
    (TDIR / "raw" / f"{img.stem}_dsocr.txt").write_text(text, encoding="utf-8")
    if "<table" in text.lower() or "<tr" in text.lower():
        g = html_to_grid(text)
        return [g] if g else []
    if "|" in text:
        g = _md_table_to_grid(text)
        if g:
            return [g]
    return _latex_tabular_to_grid(text)


ENGINES = {"ppstructure": grids_ppstructure, "rapidocr": grids_rapidocr,
           "gotocr": grids_gotocr, "dsocr": grids_dsocr}

# 表 id → 该表的扫描页文件
def _scan_pages(table_id: str) -> list[Path]:
    return sorted((TDIR / "scan").glob(f"{table_id}_p*_scan.jpg"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=list(ENGINES), default="ppstructure")
    ap.add_argument("--mode", choices=["naive", "robust", "both"], default="both")
    ap.add_argument("--table", default="")
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    gts = sorted((TDIR / "gt").glob("*.json"))
    if args.table:
        gts = [g for g in gts if g.stem == args.table]
    modes = ["naive", "robust"] if args.mode == "both" else [args.mode]
    engine = ENGINES[args.engine]

    results = []
    print(f"engine={args.engine}")
    print(f"{'table':16s} {'mode':7s} {'row_acc':>8} {'field_em':>9} {'text_hit':>9}  worst_fields")
    for gt_path in gts:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        grids: list = []
        for page_img in _scan_pages(gt["table_id"]):
            grids.extend(engine(page_img))
        # text_hit：GT 字段文本在 OCR 全文里的出现率（不管落在哪个格子）。
        # 它和 field_em 的差 = "字符识别对了但结构还原错了"的那部分——
        # 正是"分层看准确率"（字符级 vs 字段级 vs 结构级）的量化。
        from table_eval import norm
        fulltext = norm("".join(c or "" for g in grids for row in g for c in row))
        score_cols = gt.get("score_columns", gt["columns"])
        cells = [norm(r[c]) for r in gt["rows"] for c in score_cols if norm(r.get(c))]
        text_hit = sum(1 for c in cells if c in fulltext) / (len(cells) or 1)
        for mode in modes:
            pred = grids_to_rows(grids, mode)
            if args.dump:
                for p in pred:
                    print("  ", p)
            s = score(gt["rows"], pred, score_cols)
            worst = sorted(s["field_em"].items(), key=lambda kv: kv[1])[:2]
            worst_s = " ".join(f"{k}={v:.2f}" for k, v in worst)
            print(f"{gt['table_id']:16s} {mode:7s} {s['row_acc']:>8.2f} "
                  f"{s['field_em_mean']:>9.2f} {text_hit:>9.2f}  {worst_s}")
            results.append({"engine": args.engine, "table": gt["table_id"], "mode": mode,
                            "text_hit": text_hit, **s})
    out = TDIR / f"results_{args.engine}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
