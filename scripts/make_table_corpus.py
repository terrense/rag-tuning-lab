# -*- coding: utf-8 -*-
"""
================================================================================
make_table_corpus.py —— E13：表格提取"五坑"测试语料生成器
--------------------------------------------------------------------------------
表格是 RAG 里最容易翻车的部分。这里合成 5 份化验单风格的表格文件，每份
精确埋一个经典坑（面试高频）：

  t1_merged      合并单元格：大类列 rowspan 跨多行，不做 fill-down 就丢归属
  t2_crosspage   跨页表格：长表在 PDF 里跨 2 页，第 2 页没有表头
  t3_multiheader 多级表头：第一行是分组(colspan)，第二行才是叶子字段
  t4_units       数值/单位分离：同一列里 "8.5 mmol/L" 混着分离写法，
                 上标(10⁹/L)、范围(3.9–6.1)、阴/阳性混排
  t5_misalign    列错位陷阱：窄列逼长项目名换行 + 空单元格 + 右对齐数字，
                 bbox 行列聚类最容易断行/串列

为什么用"合成"而不是找真件：ground truth 是我们自己定义的（gt/*.json），
字段级 exact match 可以程序化打分——这正是"评估口径"的落点：不报笼统的
字符准确率，报 字段级EM / 行级完整率，分 clean-PDF / scan 两档。

坑注（环境教训）：最初用 Edge headless 渲 HTML→PDF，但本机的透明加密软件
会把浏览器进程写的文件变成 %TSD-Header% 密文（Python 读不了）。Python 进程
写的文件不受影响 → 全部改用 reportlab 纯 Python 生成。HTML 版仍然保留，
作为排版预览和"HTML 解析"对照臂。

产出（data/tables/）：
  gt/tN_*.json     每行 {category,item,result,unit,ref_range,flag} 的真值
  html/tN_*.html   排版预览 / HTML 解析对照
  pdf/tN_*.pdf     干净数字版（带文本层，考 pdfplumber 类数字解析）
  scan/tN_*.jpg    扫描退化版（旋转+噪声+模糊+JPEG压缩）——OCR 的战场

用法：  python scripts/make_table_corpus.py
================================================================================
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tables"
FONT = "C:/Windows/Fonts/simhei.ttf"

# ---------------------------------------------------------------------------
# 真值数据：化验单条目 (大类, 项目, 结果, 单位, 参考范围, 标记)
# ---------------------------------------------------------------------------
CBC = [  # 血常规
    ("血常规", "白细胞计数(WBC)", "6.2", "10⁹/L", "3.5–9.5", ""),
    ("血常规", "红细胞计数(RBC)", "4.8", "10¹²/L", "4.3–5.8", ""),
    ("血常规", "血红蛋白(HGB)", "128", "g/L", "130–175", "↓"),
    ("血常规", "血小板计数(PLT)", "245", "10⁹/L", "125–350", ""),
    ("血常规", "中性粒细胞百分比", "62.5", "%", "40–75", ""),
    ("血常规", "淋巴细胞百分比", "28.1", "%", "20–50", ""),
]
LIVER = [  # 肝功能
    ("肝功能", "丙氨酸氨基转移酶(ALT)", "58", "U/L", "9–50", "↑"),
    ("肝功能", "天冬氨酸氨基转移酶(AST)", "41", "U/L", "15–40", "↑"),
    ("肝功能", "总胆红素(TBIL)", "14.2", "μmol/L", "3.4–20.4", ""),
    ("肝功能", "直接胆红素(DBIL)", "3.8", "μmol/L", "0–6.8", ""),
    ("肝功能", "白蛋白(ALB)", "43.6", "g/L", "40–55", ""),
]
GLU_LIPID = [  # 血糖血脂
    ("血糖血脂", "空腹血糖(GLU)", "8.5", "mmol/L", "3.9–6.1", "↑"),
    ("血糖血脂", "糖化血红蛋白(HbA1c)", "7.8", "%", "4.0–6.0", "↑"),
    ("血糖血脂", "总胆固醇(TC)", "5.9", "mmol/L", "2.9–5.2", "↑"),
    ("血糖血脂", "甘油三酯(TG)", "2.31", "mmol/L", "0.45–1.70", "↑"),
    ("血糖血脂", "低密度脂蛋白(LDL-C)", "3.8", "mmol/L", "1.3–3.4", "↑"),
]
ELECTRO = [  # 电解质
    ("电解质", "血清钾(K)", "4.1", "mmol/L", "3.5–5.3", ""),
    ("电解质", "血清钠(Na)", "141", "mmol/L", "137–147", ""),
    ("电解质", "血清氯(Cl)", "103", "mmol/L", "99–110", ""),
    ("电解质", "血清钙(Ca)", "2.31", "mmol/L", "2.11–2.52", ""),
]
IMMUNE = [  # 免疫（定性结果，考"阳性/阴性"与数值混排）
    ("免疫检验", "乙型肝炎病毒表面抗原(HBsAg)定量检测", "0.02", "IU/mL", "<0.05", "阴性"),
    ("免疫检验", "乙型肝炎病毒表面抗体(抗-HBs)定量检测", "156.8", "mIU/mL", ">10为有免疫力", "阳性"),
    ("免疫检验", "丙型肝炎病毒抗体(抗-HCV)", "0.08", "S/CO", "<1.0", "阴性"),
    ("免疫检验", "梅毒螺旋体特异性抗体(TP-Ab)", "0.11", "S/CO", "<1.0", "阴性"),
]
ALL_PANELS = CBC + LIVER + GLU_LIPID + ELECTRO + IMMUNE

COLS = ["category", "item", "result", "unit", "ref_range", "flag"]

# SimHei 没有上标⁹/¹²字形，PDF 里用 10^9 写法（GT 保持 Unicode 上标，
# 评分器做归一化——这本身就是"单位归一化"考点的一部分）
_PDF_SUBS = {"10⁹/L": "10^9/L", "10¹²/L": "10^12/L", "–": "-"}


def _pdf_text(s: str) -> str:
    for k, v in _PDF_SUBS.items():
        s = s.replace(k, v)
    return s


# ===========================================================================
# HTML 版（预览 + HTML 解析对照臂）
# ===========================================================================
CSS = """
body { font-family: "Microsoft YaHei", SimSun, sans-serif; margin: 24px; font-size: 13px; }
h2 { text-align: center; margin: 4px 0; } .meta { display: flex; justify-content: space-between;
margin: 8px 0; font-size: 12px; } table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #333; padding: 4px 6px; } th { background: #eee; }
td.num { text-align: right; } .foot { margin-top: 10px; font-size: 11px; }
"""


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>"
            f"</head><body><h2>某某医院检验报告单</h2>"
            f"<div class='meta'><span>姓名：王某某　性别：男　年龄：54岁</span>"
            f"<span>科室：内分泌科　标本：静脉血</span><span>报告日期：2026-07-06</span></div>"
            f"<div class='meta'><b>{title}</b></div>{body}"
            f"<div class='foot'>检验者：李某　审核者：张某　本报告仅对该标本负责。</div></body></html>")


def _html_rowspan(rows: list[tuple]) -> str:
    out = ["<table><tr><th>检验大类</th><th>项目</th><th>结果</th><th>单位</th><th>参考范围</th><th>提示</th></tr>"]
    i = 0
    while i < len(rows):
        span = sum(1 for r in rows[i:] if r[0] == rows[i][0])
        for j in range(span):
            r = rows[i + j]
            cells = f"<td>{r[1]}</td><td class='num'>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td>"
            out.append(f"<tr><td rowspan='{span}'>{rows[i][0]}</td>{cells}</tr>" if j == 0 else f"<tr>{cells}</tr>")
        i += span
    out.append("</table>")
    return "".join(out)


def _html_flat(rows: list[tuple], header: bool = True) -> str:
    out = ["<table>"]
    if header:
        out.append("<tr><th>检验大类</th><th>项目</th><th>结果</th><th>单位</th><th>参考范围</th><th>提示</th></tr>")
    for r in rows:
        out.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td class='num'>{r[2]}</td>"
                   f"<td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td></tr>")
    out.append("</table>")
    return "".join(out)


# ===========================================================================
# PDF 版（reportlab platypus：SPAN 实现 rowspan/colspan）
# ===========================================================================
def _pdf_doc(path: Path, title: str, tables: list) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    if "SimHei" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("SimHei", FONT))
    h = ParagraphStyle("h", fontName="SimHei", fontSize=14, alignment=1)
    meta = ParagraphStyle("m", fontName="SimHei", fontSize=9)
    story = [Paragraph("某某医院检验报告单", h), Spacer(1, 6),
             Paragraph("姓名：王某某　性别：男　年龄：54岁　　科室：内分泌科　标本：静脉血　　报告日期：2026-07-06", meta),
             Paragraph(title, meta), Spacer(1, 6)]
    story += tables
    story += [Spacer(1, 8), Paragraph("检验者：李某　审核者：张某　本报告仅对该标本负责。", meta)]
    SimpleDocTemplate(str(path), pagesize=A4, topMargin=36, bottomMargin=36).build(story)


_BASE_STYLE = [
    ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.6, "black"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]


def _tbl(data, style, colWidths=None, rowHeights=None):
    from reportlab.platypus import Table
    return Table(data, style=_BASE_STYLE + style, colWidths=colWidths,
                 rowHeights=rowHeights, repeatRows=0)   # repeatRows=0：跨页不重复表头（t2 的坑）


HEADER6 = ["检验大类", "项目", "结果", "单位", "参考范围", "提示"]


def pdf_t1(path: Path, rows: list[tuple]) -> None:
    """rowspan：同大类只在首行写字，SPAN 竖向合并。"""
    data, style = [HEADER6], [("BACKGROUND", (0, 0), (-1, 0), "#eeeeee")]
    i, r0 = 0, 1
    while i < len(rows):
        span = sum(1 for r in rows[i:] if r[0] == rows[i][0])
        for j in range(span):
            r = rows[i + j]
            data.append([_pdf_text(rows[i][0]) if j == 0 else "",
                         _pdf_text(r[1]), r[2], _pdf_text(r[3]), _pdf_text(r[4]), r[5]])
        style.append(("SPAN", (0, r0), (0, r0 + span - 1)))
        i += span
        r0 += span
    style.append(("ALIGN", (2, 1), (2, -1), "RIGHT"))
    _pdf_doc(path, "生化+血常规联合报告（合并单元格版）", [_tbl(data, style)])


def pdf_t2(path: Path, rows: list[tuple]) -> None:
    """跨页：40 行 + 加高行距，repeatRows=0 → 第 2 页无表头。"""
    data = [HEADER6] + [[_pdf_text(c) for c in r] for r in rows]
    style = [("BACKGROUND", (0, 0), (-1, 0), "#eeeeee"),
             ("ALIGN", (2, 1), (2, -1), "RIGHT")]
    _pdf_doc(path, "全项联合报告（跨页版）",
             [_tbl(data, style, rowHeights=[22] * len(data))])


def pdf_t3(path: Path, rows: list[tuple]) -> None:
    """多级表头：首行分组(colspan)，次行叶子列。"""
    data = [["检验大类", "项目", "结果信息", "", "参考信息", ""],
            ["", "", "测定值", "单位", "参考范围", "异常提示"]]
    data += [[_pdf_text(c) for c in r] for r in rows]
    style = [
        ("SPAN", (0, 0), (0, 1)), ("SPAN", (1, 0), (1, 1)),      # 两个竖向合并
        ("SPAN", (2, 0), (3, 0)), ("SPAN", (4, 0), (5, 0)),      # 两个分组横向合并
        ("BACKGROUND", (0, 0), (-1, 1), "#eeeeee"),
        ("ALIGN", (0, 0), (-1, 1), "CENTER"),
        ("ALIGN", (2, 2), (2, -1), "RIGHT"),
    ]
    _pdf_doc(path, "检验报告（多级表头版）", [_tbl(data, style)])


def pdf_t4(path: Path, rows: list[tuple]) -> None:
    """数值/单位：偶数行结果格写成 "8.5 mmol/L"，奇数行数值与单位空多格分离。"""
    data = [["项目", "结果", "参考范围", "提示"]]
    for i, r in enumerate(rows):
        res = f"{r[2]} {_pdf_text(r[3])}" if i % 2 == 0 else f"{r[2]}    {_pdf_text(r[3])}"
        data.append([_pdf_text(r[1]), res, f"{_pdf_text(r[4])} {_pdf_text(r[3])}", r[5]])
    style = [("BACKGROUND", (0, 0), (-1, 0), "#eeeeee"),
             ("ALIGN", (1, 1), (1, -1), "RIGHT")]
    _pdf_doc(path, "生化报告（数值单位混排版）", [_tbl(data, style)])


def pdf_t5(path: Path, rows: list[tuple]) -> None:
    """列错位：窄列 + Paragraph 换行 + 空单元格 + 右对齐。"""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    cell = ParagraphStyle("c", fontName="SimHei", fontSize=9, leading=11)
    data = [["项目", "结果", "单位", "参考范围", "判定"]]
    for r in rows:
        data.append([Paragraph(_pdf_text(r[1]), cell), r[2], _pdf_text(r[3]),
                     Paragraph(_pdf_text(r[4]), cell), r[5]])
    style = [("BACKGROUND", (0, 0), (-1, 0), "#eeeeee"),
             ("ALIGN", (1, 1), (1, -1), "RIGHT")]
    _pdf_doc(path, "免疫+电解质报告（窄列换行版）",
             [_tbl(data, style, colWidths=[150, 55, 55, 130, 60])])


# ===========================================================================
# 表定义：名字 → (坑说明, 行数据, HTML生成, PDF生成)
# ===========================================================================
T2_ROWS = (ALL_PANELS * 2)[:40]
# score_columns：只对"文档上确实存在/可恢复"的字段计分。
# t4/t5 的版面上没有大类列，把 category 计进去会让满分解析器也吃 0 分。
TABLES = {
    "t1_merged": ("合并单元格：大类列 rowspan，需 fill-down 才能还原每行归属",
                  CBC + LIVER + GLU_LIPID, COLS,
                  lambda rows: _page("生化+血常规联合报告（合并单元格版）", _html_rowspan(rows)),
                  pdf_t1),
    "t2_crosspage": ("跨页表格：40 行跨 2 页，第 2 页无表头，需表头继承",
                     T2_ROWS, COLS,
                     lambda rows: _page("全项联合报告（跨页版）", _html_flat(rows)),
                     pdf_t2),
    "t3_multiheader": ("多级表头：colspan 分组行 + 叶子行，只抽最后一行会丢上级语义",
                       CBC + ELECTRO, COLS,
                       lambda rows: _page("检验报告（多级表头版）", _html_flat(rows)),
                       pdf_t3),
    "t4_units": ("数值单位分离：同格混排 vs 分离，含上标 10⁹/L、范围 3.9–6.1、阴/阳性",
                 GLU_LIPID + LIVER, ["item", "result", "unit", "ref_range", "flag"],
                 lambda rows: _page("生化报告（数值单位混排版）", _html_flat(rows)),
                 pdf_t4),
    "t5_misalign": ("列错位：超长项目名换行 + 空单元格 + 右对齐数字，考行列聚类",
                    IMMUNE + ELECTRO, ["item", "result", "unit", "ref_range", "flag"],
                    lambda rows: _page("免疫+电解质报告（窄列换行版）", _html_flat(rows)),
                    pdf_t5),
}


def degrade_to_scan(pdf_path: Path, scan_dir: Path, seed: int = 7) -> list[Path]:
    """PDF → 仿扫描 JPG：150dpi 渲染 + 轻微旋转 + 高斯噪声 + 模糊 + JPEG 压缩。"""
    import fitz
    import numpy as np
    from PIL import Image, ImageFilter

    rng = random.Random(seed)
    outs = []
    doc = fitz.open(pdf_path)
    for pno, pg in enumerate(doc):
        pix = pg.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = img.rotate(rng.uniform(-1.6, 1.6), expand=True, fillcolor=(238, 236, 230))
        arr = np.asarray(img).astype(np.int16)
        noise = np.random.default_rng(seed + pno).normal(0, 7, arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype("uint8"))
        img = img.filter(ImageFilter.GaussianBlur(0.6))
        out = scan_dir / f"{pdf_path.stem}_p{pno + 1}_scan.jpg"
        img.save(out, quality=55)
        outs.append(out)
    doc.close()
    return outs


def main() -> None:
    argparse.ArgumentParser().parse_args()
    for sub in ("gt", "html", "pdf", "scan"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    for name, (pitfall, rows, score_cols, html_fn, pdf_fn) in TABLES.items():
        (OUT / "html" / f"{name}.html").write_text(html_fn(rows), encoding="utf-8")
        gt = {"table_id": name, "pitfall": pitfall, "columns": COLS,
              "score_columns": score_cols,
              "rows": [dict(zip(COLS, r)) for r in rows]}
        (OUT / "gt" / f"{name}.json").write_text(
            json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
        pdf = OUT / "pdf" / f"{name}.pdf"
        pdf_fn(pdf, rows)
        scans = degrade_to_scan(pdf, OUT / "scan")
        print(f"{name}: {len(rows)} rows, pdf ok, {len(scans)} scan page(s)")
    print(f"\ncorpus -> {OUT}")


if __name__ == "__main__":
    main()
