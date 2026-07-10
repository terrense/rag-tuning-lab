# RAG Tuning Lab

一个**干中学**的渐进式 RAG 实验台：从最朴素的相似度检索，一步步做到**多模态**与
**GraphRAG**，每个能力都配**可运行 demo**和**可量化的实验追踪**。不是 demo 堆砌，
而是"改一个变量 → 看指标变化 → 理解它解决了哪类失败"。

> 配置驱动（`configs/*.yaml` + `--set a.b=c` 覆盖），离线建库 / 在线查询分离，
> 实验自动记录到 `experiments/`。向量库 **Chroma**。三个 LLM 按任务分工
> （`llm.py` 角色路由，yaml 一行切换）：**MiniMax M3** 多模态+生成、
> **deepseek-v4-pro** 裁判/质检、**deepseek-v4-flash** 便宜跑量（改写/出题）。

---

## 三种 RAG 范式（本项目都实现了）

| 范式 | 擅长的问题 | 实现 | 怎么跑 |
|---|---|---|---|
| **① 普通 RAG** | 局部事实问答 | 向量 + BM25 → RRF 融合 → cross-encoder 精排 | `configs/diseases.yaml` |
| **② 多模态 RAG** | 图 / 表 问答 | 文字 + 表格(Markdown) + 配图(M3视觉描述)；回答时把真实图喂回 M3 | `configs/docs.yaml` |
| **③ GraphRAG** | 多跳关系 / 全局概览 | 抽三元组 → 知识图 + 实体消歧 → 图遍历 / 社区摘要 | `rag_lab.graph_*` |

并配套多类**进阶检索技法**：三层 **query 改写**（规则 / 传统NLP / LLM）、
**CLIP 图向量 vs 描述法**对比、**CRAG 检索自纠错**、**Parent-Document / Contextual
Retrieval**、**表格/OCR 四臂基准**（见 `experiments/`）。

**生产服务层**（`rag_lab.serve`）：FastAPI 常驻服务，`/ask` `/search` `/ask/stream`(SSE)，
分阶段 tracing（延迟+token+成本）、语义缓存（命中快 ~270 倍）。见 `experiments/serving.md`。

---

## 快速开始

```powershell
# 专用 conda 环境（不要用 base）
conda env create -f environment.yml   # 或用现有 rag-tuning-lab 环境
$py = "C:/Users/Administrator/miniconda3/envs/rag-tuning-lab/python.exe"
$env:PYTHONIOENCODING = "utf-8"        # 让中文输出不乱码

# LLM 密钥放 .env（已 gitignore，切勿提交）
#   MINIMAX_API_KEY=...
#   MINIMAX_BASE_URL=https://api.minimaxi.com/v1
#   MINIMAX_MODEL=MiniMax-M3
```

---

## 用法

### ① 普通 RAG（医疗结构化语料）
```powershell
& $py -m rag_lab.ingest --config configs/diseases.yaml      # 建库
& $py -m rag_lab.query  --config configs/diseases.yaml --query "苯中毒的症状和治疗"
& $py -m rag_lab.ask    --config configs/diseases.yaml --query "..."   # 检索+生成带引用
```

### ② 多模态 RAG（PDF：文字+表格+配图）
```powershell
& $py -m rag_lab.ingest --config configs/docs.yaml          # 抽文字/表格 + M3 给图配描述
& $py -m rag_lab.ask    --config configs/docs.yaml --query "图神经网络有哪几种架构？"
#   命中配图时会把真实图片喂回 M3 做图文联合回答
```

### 热启动 REPL（避免每次重载模型，第二问起几秒）
```powershell
& $py -m rag_lab.repl --config configs/docs.yaml            # 一次加载，反复提问
& $py -m rag_lab.repl --config configs/docs.yaml --no-llm   # 只检索，最快
```

### 三层 query 改写（解决 语义鸿沟 / 意图模糊 / 历史指代）
```powershell
& $py -m rag_lab.ask --config configs/diseases.yaml --set query.llm=rewrite `
      --history "用户：我确诊了糖尿病" --query "它有哪些并发症？"
# query.rules(规则) / query.nlp(传统NLP) / query.llm(rewrite|hyde|multi) 三层可独立开关
```

### ③ GraphRAG
```powershell
& $py -m rag_lab.graph_extract --config configs/docs.yaml --keywords "maple,clip2scene,vita_clip" --per-paper 4
& $py -m rag_lab.graph_build   --embed-dedup        # 建图 + 实体消歧（字符串+embedding）
& $py -m rag_lab.graph_query   --query "和 Vita-CLIP 用同一基础模型的论文有哪些？" --hops 2
& $py -m rag_lab.graph_community --summarize --query "这批论文整体在研究哪些方向？"
```

### 实验追踪（用数字代替感觉，且带显著性检验）
```powershell
& $py -m rag_lab.experiment --config configs/diseases.yaml --label "my-run"
& $py -m rag_lab.compare_runs --a my-run --b v2-baseline        # 配对置换检验：差异是真的吗（p值）
& $py -m rag_lab.gen_eval --config configs/diseases.yaml --label "gen-run"  # 生成端：引用指标+LLM裁判
& $py -m rag_lab.evalgen --n 150 --seed 42                      # 重建自动评测集（flash出题+pro质检）
& $py -m rag_lab.clip_index --config configs/docs.yaml --build   # CLIP vs 描述法
& $py -m rag_lab.clip_index --config configs/docs.yaml --compare
type experiments\LEADERBOARD.md
```
评测集 v2：119 题（10 手写 + 109 生成，科室分层、四风格、双模型交叉质检）。
LEADERBOARD 每行带 bootstrap 95% CI。实验路线图见 `experiments/PLAN.md`。

---

## 实验结果（真实数据，详见 `experiments/`）

- **检索调参**（医疗集，`experiments/LEADERBOARD.md`）：候选池 12→50、BM25 权重 0.3→0.6，
  Recall@5 **0.30 → 0.70**，MRR 0.30 → 0.55。
- **三层 query 改写**：MRR 0.50 → 0.575，nDCG@5 0.59 → 0.72，BM25 Recall@10 → 1.00。
- **CLIP 图向量 vs 描述法**（`experiments/clip_vs_caption.md`）：描述法 **10/10**，CLIP 3/10——
  文档框架图含义在文字里，CLIP 读不了图中文字；且我们喂的是整页渲染，稀释了 CLIP 视觉信号。
- **延迟**：冷启动 ~40s 中 ~32s 是模型加载（一次性）；真正检索仅 ~2s。REPL 热启动后每问 ~2.7s。

---

## 架构（`src/rag_lab/`）

```
建库(离线): 语料 → loaders/structured/multimodal → chunking → embeddings → stores(Chroma)
查询(在线): 问题 → [query_rewrite] → 向量+BM25 → RRF(pipeline) → rerankers → [generate(MiniMax)]
评测:       metrics + experiment(追踪) ;  GraphRAG: graph_extract/build/query/community
```
核心编排在 `pipeline.py`。各模块均有详细中文注释。

## 工程要点
- **离线重、在线轻**：贵的活（embedding、图抽取）放建库；查询只剩检索+生成。
- **缓存**：embedder / cross-encoder / chunks+BM25 跨查询复用；图描述落盘缓存（断点续跑、不重复计费）。
- **容错**：单张图 caption 失败不拖垮建库；图文请求失败自动退回纯文字。
- **追踪**：每次实验记 git sha + 参数 + 多阶段指标 + 延迟，进 `LEADERBOARD.md`。

## 诚实的局限
- 评测集较小；GraphRAG 图偏稀疏、字符串消歧抓不住全部语义近义词、抽取有噪声。
- 多语言 MiniLM 对中文医疗文本偏弱（vector-only 召回约 0.20）——可换中文专用 embedding。

详细路线图见 [ROADMAP.md](ROADMAP.md)。
