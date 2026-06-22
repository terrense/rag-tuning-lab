# Roadmap

本项目是一个**干中学（learn-by-doing）的 RAG 实验台**，锚定真实医疗语料，逐级把检索能力从纯文本扩展到多模态。每一级都对应一个可运行的实验和一条要验证的"教训"。

分工：实验设计与代码由 Claude 负责，语料由作者提供。

## 现状（baseline）

两阶段检索已就绪：

```
query
  ├── vector recall (multilingual MiniLM embedding, Chroma / Milvus)
  └── BM25 recall (rank_bm25)
        └── RRF 融合
              └── cross-encoder rerank (mmarco-mMiniLMv2)
                    └── top-k
```

- 配置驱动：`configs/*.yaml`，支持 `--set a.b=c` 覆盖
- 离线评测：hit / first_rank / MRR
- 参数 sweep：`python -m rag_lab.sweep ...`
- 文档加载器目前支持 `.pdf` / `.txt` / `.md`

## 分级路线图

| 级别 | 主题 | 内容 | 状态 |
|------|------|------|------|
| **L0** | 结构化导入 | `.json` / `.xlsx` loader。把记录 verbalize 成可检索文本，字段保留为 metadata 供过滤。语料：`diseases_clean.json`（5942 条疾病记录：疾病名称/科室/症状/病因/检查项目…）+ 分科室 xlsx 疾病表 | ⬜ 进行中 |
| **L1** | 生成 + 引用 | 在检索结果上做 LLM 生成，带可溯源引用（Claude API，最新 Opus/Sonnet 模型；key 待配置） | ⬜ |
| **L2** | 表格 | xlsx 表格检索；后续扩展到 PDF 化验单表格 | ⬜ |
| **L3** | 跨页 / 层级 | 教材类长文档（诊断学 PDF）的层级切分与跨页上下文。等未加密 PDF | ⛔ 阻塞 |
| **L4** | OCR | 扫描件 PDF 的 OCR 接入 | ⛔ 阻塞 |
| **L5** | 多模态 | 心电 / 影像报告（含图）的图文联合检索与生成 | ⛔ 阻塞，待素材 |

## 计划逐步引入的进阶 RAG 技法

随级别推进，挑选合适的引入并做 A/B 对比（都纳入离线评测）：

- **查询侧**：query rewriting / HyDE、多查询扩展、子问题分解（decomposition）
- **检索侧**：RRF 已做；继续 MMR 去冗余、metadata 过滤路由、parent-document / small-to-big、句子窗口检索
- **重排与压缩**：cross-encoder（已做）、上下文压缩 / 抽取式精排、LLM rerank
- **结构化**：知识图谱 / GraphRAG、表格转结构化查询
- **生成与校验**：引用对齐、self-RAG / CRAG（检索质量自评与纠错）、答案 groundedness 校验
- **多模态**：图文 embedding（CLIP 类）、图表理解、版面分析（layout-aware）
- **评测**：从 hit/MRR 扩展到 faithfulness / answer relevance 等生成质量指标

> 进阶技法按"边学边干"的节奏引入，不一次堆砌；每引入一个都要能在评测上看到它解决了哪类失败。

## 环境约束

- C 盘有透明加密（`%TSD-Header%`），碰过的文件会变密文 → **语料与代码均放在 F 盘**。
- 用专用 conda 环境 `rag-tuning-lab` 运行，不要用 base。
