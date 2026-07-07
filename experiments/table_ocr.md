# E13 — 表格结构化提取基准（五坑 × 四臂）

> 面试问的是"表格 OCR 踩过哪些坑、准确率怎么报"。这个实验把那套说辞变成
> **自己实验室里可复现的数字**：合成 5 份各埋一个经典坑的化验单，真值自定义，
> 用 **field-level EM + row-level 完整率**（不报笼统字符准确率）横评四条提取路线。

## 语料（`scripts/make_table_corpus.py` → `data/tables/`）

5 份化验单风格表格，每份精确埋一个坑；每份三种载体：带文本层 PDF、
仿扫描 JPG（旋转+高斯噪声+模糊+JPEG 压缩）、HTML（对照）。真值 `gt/*.json`
逐单元格定义 → 字段级 EM 可程序化打分。

| 表 | 坑 | 说明 |
|---|---|---|
| t1_merged | 合并单元格 | 大类列 rowspan 跨多行，不 fill-down 就丢每行归属 |
| t2_crosspage | 跨页无表头 | 40 行跨 2 页，第 2 页没有表头 |
| t3_multiheader | 多级表头 | 分组行(colspan) + 叶子行，只抽末行丢上级语义 |
| t4_units | 数值单位分离 | "8.5 mmol/L" 混排、上标 10⁹/L、范围 3.9–6.1、阴/阳性 |
| t5_misalign | 列错位 | 窄列逼长项目名换行 + 空单元格 + 右对齐数字 |

## 评估口径（关键）

- **field_em**：逐单元格 exact match，做单位归一化（10⁹→10^9、µ→μ、–→-、全半角）。
- **row_acc**：一行全部计分字段都对才算这行对（数值/单位错位会立刻炸）。
- **text_hit**：GT 文本在 OCR 全文里的命中率（不管落在哪个格）——它与 field_em 的
  差，就是"**字符识别对了但结构还原错了**"的量化，正是"分层看准确率"那句话的落点。

## 四条提取路线

| 臂 | 路线 | 输入 |
|---|---|---|
| **pdfplumber** | 文本层数字解析（naive vs robust 修复） | 带文本层 PDF |
| **RapidOCR** | 裸文字框 + 自研 y/x bbox 聚类重建行列 | 扫描 JPG |
| **PP-StructureV3** | 版面分析 + 表格识别 → HTML 结构 | 扫描 JPG |
| **GOT-OCR2.0** | 端到端 VLM → LaTeX/HTML（GPU） | 扫描 JPG |

robust 修复 = fill-down（合并单元格）+ 跨页表头继承 + 多级表头拼接 +
数值单位拆分，四条臂共用同一套 `grids_to_rows`（对照才干净）。

## 结果一：带文本层 PDF（pdfplumber，naive → robust）

| 表 | naive row_acc | robust row_acc |
|---|:---:|:---:|
| t1_merged | 0.19 | **1.00** |
| t2_crosspage | 0.97 | **1.00** |
| t3_multiheader | 1.00 | 1.00 |
| t4_units | 0.00 | **1.00** |
| t5_misalign | 1.00 | 1.00 |

**结论**：数字 PDF 的翻车点在**语义还原**（归属、拆分），不在结构识别——文本层
保住了单元格边界，t3/t5 天然免疫。修复到位就能满分。

## 结果二：扫描件三臂对照（robust 模式，field_em / row_acc）

| 表 | 坑 | RapidOCR | PP-StructureV3 | GOT-OCR2.0 |
|---|---|:---:|:---:|:---:|
| t1_merged | 合并单元格 | 0.00 / 0.00 | 0.78 / 0.75 | 0.93 / 0.56 |
| t2_crosspage | 跨页无表头 | 0.43 / 0.10 | 0.73 / 0.50 | 0.94 / 0.68 |
| t3_multiheader | 多级表头 | 0.00 / 0.00 | 0.87 / 0.80 | 0.97 / 0.80 |
| t4_units | 数值单位 | 0.28 / 0.00 | 0.48 / 0.30 | 0.84 / 0.50 |
| t5_misalign | 列错位 | 0.05 / 0.00 | 0.70 / 0.50 | **1.00 / 1.00** |

所有臂的 **text_hit 都在 0.86–1.00**——字符都认得出，差距全在结构还原。

## 三个可直接讲的结论

1. **"不报笼统准确率"有了数据背书**：RapidOCR 字符命中 0.86–0.98，但字段级
   0.00–0.43、行级几乎全灭。裸 OCR 盒子 + 自研行列聚类在 t1/t3/t5 上被
   标题污染列聚类、相邻单元格并框、换行拆行彻底打崩——**字符准确率高不代表业务可用**。

2. **结构化管线 vs 端到端 VLM 各有胜负**：
   - PP-StructureV3（版面+表格识别）是稳的中文基线，t1/t3 靠显式表格结构拿高分。
   - GOT-OCR2.0（VLM 直出 LaTeX）field_em 普遍更高，**t5 列错位满分**——
     VLM 对"视觉上属于同一行"的理解强于 bbox 聚类；但它会**幻觉结构**
     （t1 把提示列误读成 multirow ↓，row_acc 反被拖到 0.56）。
   - 取舍：要**稳定结构**用 PP-Structure，要**抗版面畸变**用 VLM，生产上可
     VLM 出结构 + 规则校验兜底（数值范围/单位白名单）。

3. **修复层是跨引擎复用的杠杆**：同一套 fill-down / 表头继承 / 数值单位拆分，
   在 pdfplumber 上把 t1 0.19→1.00、t4 0.00→1.00，在 OCR 网格上同样生效
   （GOT t4 0.42→0.84）。**OCR 只负责把像素变成网格，语义还原是可移植的后处理**。

## 工程坑（面试可聊）

- **透明加密**：Edge headless 渲的 PDF 被本机加密软件写成 `%TSD-Header%` 密文
  → 改用 reportlab 纯 Python 生成。
- **PaddlePaddle 3.x Windows**：PIR + oneDNN 执行器抛
  `ConvertPirAttribute2RuntimeAttribute` → `enable_mkldnn=False` 绕过（只影响
  CPU 推理速度，不影响结果）。
- **GOT LaTeX 解析**：`\multirow` 续行留空占位 `&`，不能自动插值（会整行右移）；
  数学模式 `\(...\)` / `\mathrm` 要还原成 GT 文本形态；单元格内嵌套 tabular
  （t5 换行）要先压平。

## 复现

```powershell
$py  = "C:/Users/Administrator/miniconda3/envs/rag-tuning-lab/python.exe"   # 主环境
$ocr = "C:/Users/Administrator/miniconda3/envs/ocr-lab/python.exe"          # OCR 环境
& $py  scripts/make_table_corpus.py           # 生成语料 + 真值
& $py  scripts/table_eval.py                  # pdfplumber 臂（数字 PDF）
& $ocr scripts/ocr_table_eval.py --engine rapidocr
& $ocr scripts/ocr_table_eval.py --engine ppstructure
& $env:HF_HOME="F:/hf_cache"; & $ocr scripts/ocr_table_eval.py --engine gotocr  # GPU
```

## 待办

- DeepSeek-OCR 臂（权重已下 `F:/hf_cache`）——与 GOT 同为 VLM 路线，补第四条扫描臂。
- 把最佳表格提取接进 RAG：表格转"Markdown（喂 LLM）+ JSON（字段校验）"双份，
  接 L2 表格问答评测。
