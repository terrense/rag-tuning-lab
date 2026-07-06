# 实验总计划（P0 → P4）

目标：把本项目从"能跑的 demo"推到"每个设计决策都有数据支撑"的生产级实验台。
每个实验都有：假设 → 单一变量 → 指标 → 显著性判定（`rag_lab.compare_runs`）。

模型分工（`llm.py` 角色路由，yaml 里 `llm.roles.*` 可换）：
**deepseek-v4-flash** 便宜跑量（出题/改写/初筛）；**deepseek-v4-pro** 难任务
（judge/质检/生成对比）；**MiniMax-M3** 多模态（唯一吃图的）+ 生成端对标 pro。

## P0 基建 —— 让实验结果可信 ✅ 本轮完成

| # | 内容 | 状态 |
|---|---|---|
| E0 | 评测集 10 → ~150 题：科室分层采样 → flash 出题（四风格轮转）→ 规则校验 → pro 质检（出题者≠质检者）→ `eval_queries_diseases_v2.yaml` | ✅ |
| E0.5 | `bootstrap_ci` + `paired_permutation_test`（metrics.py）；逐题×逐阶段指标进 runs.jsonl；`compare_runs` CLI 出 p 值；LEADERBOARD 加 95% CI 列 + eval set 列 | ✅ |
| E0.7 | `gen_eval.py`：引用合法率/精确率/召回（程序算）+ faithfulness/relevance（LLM-as-judge，judge≠generator）+ 诚实弃答率 | ✅ |

依据（为什么 N=10 不行）：Recall@5=0.70 在 N=10 时 bootstrap 95% CI 是
**[0.40, 1.00]**（宽 0.60）；N=150 收窄到 ~±0.08。旧 LEADERBOARD 上
0.70 vs 0.60 的"提升"，配对检验 p≈0.25 —— 是噪声。

## P1 检索质量 —— 预期数据最多的一批

| # | 假设 | 变量 | 判定指标 | 依赖 |
|---|---|---|---|---|
| E1 | 中文专用 embedding 大幅提升向量路（现 vector-only R@5≈0.20） | MiniLM-多语 vs **bge-small-zh-v1.5**（CPU 可跑）vs bge-m3（等 GPU） | vector_only 与 hybrid_rerank 的 R@5/MRR + p 值；顺带重扫 bm25_weight（向量变强后最优权重应回落） | 无 |
| E2 | 更强 reranker 值回延迟成本 | mmarco-mMiniLM vs bge-reranker-v2-m3（GPU）vs 无精排 vs LLM 精排（flash 逐条打分） | R@5 / p50 延迟 / 每查询成本 三维曲线 | v2 基线 |
| E3 | 检索粒度与生成上下文可以解耦 | chunk 尺寸 sweep；parent-document（小块检索、大块喂 LLM）；句子窗口 | 检索 R@5 + 生成端 faithfulness 同时看 | E0.7 |
| E4 | metadata 过滤能提精度降延迟 | flash 判科室 → Chroma where 过滤 vs 全库 | R@5、p50、错判科室时的降级表现 | 无 |
| E5 | query 改写这种轻任务 flash 够用 | {rewrite, hyde, multi} × {minimax, flash} 矩阵 | 检索指标 + 改写延迟 + token 成本 | 无 |

## P2 生成与校验

| # | 假设 | 设计 |
|---|---|---|
| E6 | 生成质量 M3 ≈ pro（价差之下谁划算） | 同一批检索结果，`llm.roles.generate` 各跑一遍；**双向交叉裁判**（pro 评 M3 用 M3 评 pro 不行——裁判也要换，用第三方位随机化+两向平均去偏）；硬指标 citation precision 不受裁判偏差影响 |
| E7 | CRAG-lite 降幻觉 | flash 给检索结果打相关分，低于阈值 → 触发改写重检或诚实拒答；指标：faithfulness、abstain_correct、幻觉率 |
| E8 | 上下文压缩省 token 不掉质量 | 抽取式压缩 vs 全文；token 成本 vs faithfulness/relevance |

## P3 多模态深化

| # | 假设 | 设计 |
|---|---|---|
| E9 | 之前 CLIP 输给 caption 是因为喂了整页渲染（信号稀释） | PyMuPDF 版面块检测裁剪图区域 → 重跑 clip_vs_caption；**验证或推翻自己的历史结论** |
| E10 | 表格问答需要独立评测 | 20 题表格评测集；Markdown 表 vs 行级切分 vs verbalize |
| E11 | caption 法优势有边界：图表型语料上 CLIP 应回血 | ChartQA test 子集（已下载 `data/external/chartqa/`）抽 200 图建库，重跑 A/B；给结论画适用边界 |
| E12 | （等 GPU）ColPali 式页面向量 | 5070 显存空闲时再做 |

### E13 表格结构化提取基准（2026-07-06 立项，第一臂已出数）

**语料**：`scripts/make_table_corpus.py` 合成 5 份化验单，每份精确埋一个坑，
真值自定义（`data/tables/gt/`）→ 字段级 EM 可程序化打分。三种载体：
带文本层 PDF（考数字解析）、扫描退化 JPG（考 OCR）、HTML（对照）。
坑注：Edge 渲的 PDF 被本机透明加密软件写成密文 → 改用 reportlab 纯 Python 生成。

**评估口径**（不报笼统字符准确率）：field_em（字段级，含单位归一化：
10⁹/10^9、µ/μ、–/-）+ row_acc（行级 6 字段全对）。

**第一臂结果**：pdfplumber 数字解析，naive vs robust（+fill-down、
+跨页表头继承、+数值单位拆分）：

| 表 | 坑 | naive row_acc | robust row_acc |
|---|---|---|---|
| t1_merged | 合并单元格 | 0.19 | **1.00** |
| t2_crosspage | 跨页无表头 | 0.97 | **1.00** |
| t3_multiheader | 多级表头 | 1.00 | 1.00 |
| t4_units | 数值单位混排 | 0.00 | **1.00** |
| t5_misalign | 窄列换行 | 1.00 | 1.00 |

结论①：数字 PDF 的翻车点在**语义还原**（归属、拆分），不在结构识别——
文本层保住了单元格边界，t3/t5 天然免疫。
结论②：t3/t5 是专门留给 OCR 臂的（bbox 行列聚类才会断行串列）——
预期 OCR 臂的失败模式与数字解析**互补**，这正是"评估口径"故事的下半场。

**OCR 臂（待跑，模型已下/环境已建）**：ocr-lab conda env。
候选：PP-StructureV3（中文表格事实标准）、RapidOCR（轻基线）、
GOT-OCR2.0（端到端 VLM，权重 1.4G）、DeepSeek-OCR（3B VLM）、
MinerU（整管线参照，可后补）。统一在 `data/tables/scan/*.jpg` 上跑同一评分器。

## P4 工程化（生产级故事）

FastAPI 服务（模型常驻 + 流式）→ docker-compose → pytest + smoke eval 进 CI
→ locust 压测 p50/p99/QPS → 延迟分解瀑布（embed/检索/精排/首token）。

## 数据资产

| 位置 | 内容 | 用途 |
|---|---|---|
| `data/files/数据集/diseases_clean.json` | 5942 条疾病记录 | 主语料 |
| `data/external/cMedQA2/` | 10万真实医疗问题 + 20万答案 | 真实用户问法反哺评测集；困难负例 |
| `data/external/chartqa/chartqa_test.parquet` | ChartQA test（含图） | E11 |

## GPU 排队区（显存空了按序跑）

1. bge-m3 embedding（E1 第三臂）
2. bge-reranker-v2-m3（E2）
3. ColPali / ColQwen 页面向量（E12）
