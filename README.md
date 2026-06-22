# RAG Tuning Lab

这个项目是一个 RAG 调参实验台。它也是一个**干中学**的渐进式项目，目标是把检索能力从纯文本一步步扩展到多模态——详见 [ROADMAP.md](ROADMAP.md)。

你主要用一个配置：

```powershell
configs/play.yaml
```

它默认使用：

- 向量库：Chroma
- Embedding：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Reranker：`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- 召回：Vector + BM25 hybrid search

## 1. 放你的资料

把资料放到：

```powershell
data/docs/
```

支持：

- PDF：`.pdf`
- 文本：`.txt`
- Markdown：`.md`、`.markdown`

要求：

- PDF 最好是可复制文字的 PDF；扫描版图片 PDF 暂时不会 OCR。
- 文件名尽量有意义，例如 `rag_chunking_notes.pdf`。
- 初次实验建议先放 3-10 个文件，别一上来塞几百个。
- `data/docs/` 默认被 git 忽略，适合放私人资料。

放完后先检查系统看见了哪些文件：

```powershell
python -m rag_lab.sources --config configs/play.yaml
```

## 2. 建库和查询

```powershell
conda activate rag-tuning-lab
python -m rag_lab.ingest --config configs/play.yaml
python -m rag_lab.query --config configs/play.yaml --query "你的问题"
```

也可以继续用内置面试题：

```powershell
python -m rag_lab.query --config configs/play.yaml --query-id q_rerank
```

如果你想只检索自己放到 `data/docs/` 的资料，不混入内置面试卡片：

```powershell
python -m rag_lab.ingest --config configs/play.yaml --set source.include_interview_cards=false
python -m rag_lab.query --config configs/play.yaml --query "你的问题" --set source.include_interview_cards=false
```

输出里重点看：

- `source`：命中的文档或知识卡 id
- `file`：如果来自 `data/docs/`，会显示文件路径和 PDF 页码
- `vector`：向量召回分数
- `bm25`：关键词召回分数
- `rerank`：reranker 分数
- `ranks`：在不同阶段的排名

## 3. 对比实验

对比 rerank 是否有用：

```powershell
python -m rag_lab.query --config configs/play.yaml --query-id q_rerank --set rerank.mode=none
python -m rag_lab.query --config configs/play.yaml --query-id q_rerank --set rerank.mode=cross_encoder
```

跑参数网格：

```powershell
python -m rag_lab.sweep --config configs/play.yaml --query-id q_rerank --vary chunking.chunk_size=240,360,520 --vary retrieval.candidate_k=6,12 --vary rerank.mode=none,cross_encoder
```

看这些指标：

- `hit`：是否命中 expected source
- `first_rank`：正确资料第一次出现的排名
- `mrr`：正确资料越靠前越高
- `top_sources`：最终排在前面的来源

## 4. 最值得调的参数

`chunking.chunk_size`  
每个 chunk 多长。太小容易丢上下文，太大容易引入噪声。

`chunking.chunk_overlap`  
相邻 chunk 重叠多少。大一点更不容易切断答案，但重复内容会变多。

`retrieval.candidate_k`  
第一阶段召回多少候选给 reranker。太小可能漏答案，太大更慢。

`retrieval.top_k`  
最终返回多少条。太少信息不足，太多噪声变大。

`retrieval.hybrid`  
是否启用向量召回 + BM25 关键词召回。

`retrieval.vector_weight`  
语义召回权重。概念型问题可以高一点。

`retrieval.bm25_weight`  
关键词召回权重。版本号、错误码、字段名、专有名词场景可以高一点。

`rerank.mode`  
可选：`none`、`bm25`、`overlap`、`cross_encoder`。重点对比 `none` 和 `cross_encoder`。

`rerank.weight`  
最终排序里 rerank 占比。越高越相信 reranker。

## 5. Milvus 版本

同样的神经模型，向量库换成 Milvus Lite：

```powershell
python -m rag_lab.ingest --config configs/play_milvus.yaml
python -m rag_lab.query --config configs/play_milvus.yaml --query "你的问题"
```

Docker Milvus Standalone 配置在：

```powershell
infra/milvus/docker-compose.yml
```

启动：

```powershell
.\scripts\start_milvus.ps1
```

## 6. 面试练习模式

```powershell
python -m rag_lab.interview_game --config configs/play.yaml --rounds 5
```

每轮先自己回答，再看 retrieval trace。重点不是背答案，而是看检索链路哪里成功、哪里失败。
