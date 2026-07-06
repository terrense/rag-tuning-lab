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
        # 扫描件有旋转 → 打开方向分类；unwarp 对平扫没必要
        _PP_PIPE = PPStructureV3(use_doc_orientation_classify=True, use_doc_unwarping=False)
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


ENGINES = {"ppstructure": grids_ppstructure, "rapidocr": grids_rapidocr}

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
