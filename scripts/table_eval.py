# -*- coding: utf-8 -*-
"""
================================================================================
table_eval.py —— E13：表格结构化提取评测（第一臂：pdfplumber 数字解析）
--------------------------------------------------------------------------------
评估口径（面试的核心答法——不报笼统字符准确率）：
  field_em   字段级 exact match：GT 的每个单元格，抽取结果是否完全一致
             （做归一化后比较：全半角/上标/破折号/空白——单位归一化本身是考点）
  row_acc    行级完整率：一行 6 个字段全对才算这行对（数值+单位错位会立刻炸）

两种解析模式对照（同一个解析器，开/关三个修复），验证五个坑各打崩什么：
  naive   逐页 extract_table，不做任何修复
  robust  + 合并单元格 fill-down（修 t1）
          + 跨页表头继承（修 t2：第 2 页无表头，沿用上一页列映射）
          + 多级表头拼接（修 t3：分组行×叶子行 联合成列名）

用法：
    python scripts/table_eval.py                 # 两种模式全跑，打印对照表
    python scripts/table_eval.py --mode robust --table t2_crosspage --dump
================================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TDIR = ROOT / "data" / "tables"
COLS = ["category", "item", "result", "unit", "ref_range", "flag"]

# 列名（含多级表头拼接后的形态）→ 语义字段
HEADER_MAP = {
    "检验大类": "category", "项目": "item", "结果": "result", "测定值": "result",
    "结果信息/测定值": "result", "单位": "unit", "结果信息/单位": "unit",
    "参考范围": "ref_range", "参考信息/参考范围": "ref_range",
    "提示": "flag", "异常提示": "flag", "参考信息/异常提示": "flag", "判定": "flag",
}


# ---------------------------------------------------------------------------
# 归一化：上标/全角/破折号/空白 —— 不归一化，字段EM会被"形近字符"淹没
# ---------------------------------------------------------------------------
_SUPS = {"⁹": "^9", "¹²": "^12", "¹": "^1", "²": "^2"}


def norm(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s)
    for k, v in _SUPS.items():            # 必须在 NFKC 之前：NFKC 会把 ⁹ 拍平成 9，上标信息就丢了
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("µ", "μ")               # U+00B5 MICRO SIGN vs U+03BC 希腊 mu：字体/提取器不一致
    s = s.replace("–", "-").replace("—", "-").replace("~", "-")
    return re.sub(r"\s+", "", s).strip()


# ---------------------------------------------------------------------------
# 抽取：pdfplumber 逐页表格 → 行 dict（naive / robust 两模式）
# ---------------------------------------------------------------------------
def _map_header(cells: list[str | None]) -> dict[int, str] | None:
    """一行表头 → {列下标: 语义字段}。识别不出一半以上列名就不算表头。"""
    m = {}
    for idx, c in enumerate(cells):
        key = HEADER_MAP.get((c or "").strip())
        if key:
            m[idx] = key
    return m if len(m) >= 3 else None


def grids_to_rows(grids: list[list[list[str | None]]], mode: str) -> list[dict]:
    """通用：若干"单元格网格"（每页/每表一个）→ 行 dict 列表。

    pdfplumber 臂和 OCR 臂共用这一段（表头识别/列映射/三个 robust 修复），
    这样两臂的差异就只剩"网格是怎么来的"——对照才干净。
    """
    rows: list[dict] = []
    colmap: dict[int, str] | None = None       # 跨页继承的列映射（robust）
    last_cat = ""                              # fill-down 记忆（robust）
    for table in grids:
        if not table:
            continue
        # --- 表头识别 ---
        hdr = _map_header(table[0])
        body_start = 1 if hdr else 0
        if mode == "robust" and len(table) > 1:
            # 多级表头修复：分组行(row0) × 叶子行(row1) 拼成 "分组/叶子" 再识别。
            # row0 因 colspan 会有空洞（识别不满 3 列），拼接后叶子列名就齐了。
            joined = []
            for i in range(len(table[1])):
                parent = (table[0][i] or "").strip() if i < len(table[0]) else ""
                leaf = (table[1][i] or "").strip()
                joined.append(f"{parent}/{leaf}" if parent and leaf and parent != leaf
                              else (leaf or parent))
            hdr2 = _map_header(joined)
            if hdr2 and len(hdr2) > len(hdr or {}):
                hdr, body_start = hdr2, 2
        if hdr:
            colmap = hdr
        elif mode == "naive" or colmap is None:
            # 没认出表头：naive 把 row0 当表头消费掉、其余按位置硬塞 COLS 顺序
            # （t2 第 2 页的真实翻车方式）；robust 则继承上一页的 colmap。
            colmap = {i: c for i, c in enumerate(COLS)}
            body_start = 1 if mode == "naive" else 0
        # --- 数据行 ---
        for cells in table[body_start:]:
            row = {c: "" for c in COLS}
            for idx, field in colmap.items():
                if idx < len(cells):
                    row[field] = (cells[idx] or "").replace("\n", "")
            if mode == "robust":
                if row["category"]:
                    last_cat = row["category"]
                else:                            # 合并单元格：空 → 沿用上一行大类
                    row["category"] = last_cat
                _split_value_unit(row)           # 数值/单位混排修复（t4）
            rows.append(row)
    return rows


def extract_pdf(pdf_path: Path, mode: str) -> list[dict]:
    """pdfplumber 臂：逐页 extract_table 得到网格，交给通用逻辑。"""
    import pdfplumber

    grids = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                grids.append(table)
    return grids_to_rows(grids, mode)


# 单个数值 + 紧跟单位（可有可无空格）。数值不含范围号，避免误拆 "3.9-6.1"。
_VAL_UNIT = re.compile(r"^([<>]?\d+(?:\.\d+)?)\s*([^\d\s].*)$")


def _split_value_unit(row: dict) -> None:
    """t4 修复：结果格里 "8.5 mmol/L" / "8.5mmol/L" → result=8.5, unit=mmol/L；
    参考范围尾巴上的单位一并剥掉（"3.9-6.1 mmol/L" → "3.9-6.1"）。
    OCR 臂里数学清洗会吃掉空格，所以拆分不能依赖空格。"""
    if not row.get("unit"):
        m = _VAL_UNIT.match((row.get("result") or "").strip())
        if m:
            row["result"], row["unit"] = m.group(1), m.group(2).strip()
    unit = (row.get("unit") or "").strip()
    ref = (row.get("ref_range") or "").strip()
    if unit and ref.endswith(unit):        # 参考范围尾部重复的单位剥掉
        row["ref_range"] = ref[: -len(unit)].strip()


# ---------------------------------------------------------------------------
# 打分：GT 行 ↔ 预测行 按 item 相似度贪心对齐，再算字段EM / 行完整率
# ---------------------------------------------------------------------------
def _sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    ga = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
    gb = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    return len(ga & gb) / max(1, min(len(ga), len(gb)))


def score(gt_rows: list[dict], pred_rows: list[dict], gt_cols: list[str]) -> dict:
    used: set[int] = set()
    field_hit = {c: 0 for c in gt_cols}
    row_hit = 0
    for g in gt_rows:
        best, best_s = None, 0.35                 # 相似度太低就当没抽到这行
        for j, p in enumerate(pred_rows):
            if j in used:
                continue
            s = _sim(g["item"], p.get("item", ""))
            if s > best_s:
                best, best_s = j, s
        if best is None:
            continue
        used.add(best)
        p = pred_rows[best]
        ok = 0
        for c in gt_cols:
            if norm(g.get(c)) == norm(p.get(c)):
                field_hit[c] += 1
                ok += 1
        if ok == len(gt_cols):
            row_hit += 1
    n = len(gt_rows)
    return {
        "n_rows": n,
        "row_acc": row_hit / n,
        "field_em": {c: field_hit[c] / n for c in gt_cols},
        "field_em_mean": sum(field_hit.values()) / (n * len(gt_cols)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive", "robust", "both"], default="both")
    ap.add_argument("--table", default="", help="只评某一张表（如 t2_crosspage）")
    ap.add_argument("--dump", action="store_true", help="打印抽取出的行，调试用")
    args = ap.parse_args()

    gts = sorted((TDIR / "gt").glob("*.json"))
    if args.table:
        gts = [g for g in gts if g.stem == args.table]
    modes = ["naive", "robust"] if args.mode == "both" else [args.mode]

    print(f"{'table':16s} {'mode':7s} {'row_acc':>8} {'field_em':>9}  worst_fields")
    for gt_path in gts:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        pdf = TDIR / "pdf" / f"{gt['table_id']}.pdf"
        for mode in modes:
            pred = extract_pdf(pdf, mode)
            if args.dump:
                for p in pred:
                    print("  ", p)
            s = score(gt["rows"], pred, gt.get("score_columns", gt["columns"]))
            worst = sorted(s["field_em"].items(), key=lambda kv: kv[1])[:2]
            worst_s = " ".join(f"{k}={v:.2f}" for k, v in worst)
            print(f"{gt['table_id']:16s} {mode:7s} {s['row_acc']:>8.2f} {s['field_em_mean']:>9.2f}  {worst_s}")


if __name__ == "__main__":
    main()
