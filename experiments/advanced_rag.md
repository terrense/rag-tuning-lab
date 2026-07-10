# 进阶 RAG 技法（Track B）：CRAG / Parent-Document / Contextual Retrieval

面试高频 + 生产有用的三个技法，每个都在 v2 评测集上量化，**不吹好、只报数**。

## 1. CRAG-lite（检索自纠错 / Corrective RAG）

**动机**：普通 RAG 检索到不相关的块，生成器还是硬编答案（幻觉）。CRAG 先自评
检索质量，再决定 proceed（照常）/ refine（改写重检）/ abstain（诚实拒答）。

**实现**（`crag.py`）：用便宜的 deepseek-flash 给 top-k 每块打相关性分（一次调用），
按阈值走三档。评估器 ≠ 生成器，成本低。

**结果（v2, N=119）——一个诚实的权衡，不是一味叫好：**

| 指标 | 值 | 解读 |
|---|---|---|
| 原始检索成功 | 84/119 | 金标文档在 top-5 |
| 真失败时正确拒答 | 23/35 = **0.66** | 检索真失败时，2/3 能避免硬编幻觉 ✓ |
| 改写救回金标 | 1 | 改写重检把召回失败的救回来（有限） |
| **能答却误拒答** | 21/84 = **0.25** | 四分之一可答的题被过度拒绝 ✗ |

**核心洞见**：CRAG 的价值（真失败时不幻觉）伴随代价（可答题被过度拒答）。而
**便宜的 flash 评估器对"症状导向"医疗题过于严苛**——症状描述和疾病记录它连不上，
把金标块打成低分 → 误拒答率高达 25%。

**这是"拒答的精确率/召回率权衡"**：想少幻觉就会多误拒。调节杠杆：
①换更强的评估器（pro，贵但准）②降 abstain 阈值（少误拒、但漏掉真失败）。
生产上取决于业务对"宁可不答"vs"宁可乱答"的偏好——医疗场景通常偏保守（可接受高拒答）。

## 2. Parent-Document Retrieval（small-to-big）

**动机**：检索要小块（语义聚焦、命中准），生成要大块（上下文完整、答案全）。
Parent-Document 两头都要：**小块检索命中，喂 LLM 时扩展成整篇父文档**。

**实现**（`parent_doc.py`）：纯查询期后处理，不改建库/不重 embedding。命中 chunk
→ 按 source_id 去重 → 拼回该文档所有 chunk（重叠感知去重）→ 喂 LLM。
实测上下文从 1490 字（chunk）→ 3176 字（parent），补全了病因/治疗等被切散的字段。

**A/B 结果（v2 子集 N=25，gen_eval，判官 deepseek-pro，判官看的是"答案实际生成
时用的那份上下文"）**：见下方表。结论：**在本语料上几乎无增益**——faithfulness
持平（4.20 vs 4.25，置信区间大幅重叠），relevance 略升（4.84→4.96）。

## 3. Contextual Retrieval（Anthropic 2024 分块增强）

**动机**：孤立的块脱离原文会"失去上下文"（"治疗以手术为主"——哪个病？）。
建库时让 LLM 给每块生成一句上下文前缀，拼上再 embedding，让孤块重新可被定位。

**实现**（`contextual.py`）：flash 逐块生成上下文、落盘缓存、全局封顶。接进
`ingest_config`（`chunking.contextual=true`）。

**机制验证（跑了 5 个"孤块"）**：一个只剩 `【症状】` 的块，contextual 给它加上了
`介绍急性呼吸窘迫综合征的症状、治疗和检查` 的前缀——正是让孤块重新可被定位。
技法工作正常（`storage/contextual_cache.json` 有缓存）。
> 踩坑：flash 是推理模型，`max_tokens=80` 会被 `<think>` 吃光、剥离后为空 → 提到 400。

**诚实的成本/价值判断**：本项目疾病语料已用 `prepend_title` 把病名塞进每块，
且每条记录本身就小——contextual 的边际收益预期很低。全量是 27468 次调用，
**为一个大概率边际的结果烧 2.7 万次调用不成比例**，故只做机制验证，量化全量 A/B
记为带 repro 的 future work（`--set chunking.contextual=true` 即可跑）。它的收益
在"块会真丢上下文"的语料（长文档被切散、多论文 PDF）才明显。

## 生成端 A/B 表（gen_eval，N=25，判官 deepseek-pro）

| 配置 | citation_precision | faithfulness | relevance | gold_cited |
|---|---|---|---|---|
| chunk（基线） | 0.335 | 4.20 [3.8, 4.56] | 4.84 | 1.00 |
| parent-document | 0.293 | 4.25 [3.92, 4.58] | 4.96 | 0.93 |

## 贯穿三个技法的一个洞见

Parent-Document 和 Contextual Retrieval 在本语料上**都几乎无增益**——不是技法没用，
是**这个语料的块本来就不丢上下文**（记录小 + prepend_title）。它们的价值在"长文档
被切散、块失去兄弟上下文"的场景。**知道一个技法什么时候该用、什么时候是过度设计，
比会实现它更重要**——这正是"改一个变量→看指标→理解它解决哪类失败"的方法论收益。

## 复现

```powershell
$py = "C:/Users/Administrator/miniconda3/envs/rag-tuning-lab/python.exe"
& $py -m rag_lab.crag --config configs/diseases.yaml --eval --set paths.eval_queries=data/eval_queries_diseases_v2.yaml
& $py -m rag_lab.gen_eval --config configs/diseases.yaml --set generation.parent_document=true --label gen-parent
# contextual: 建库时 --set chunking.contextual=true --set chunking.contextual_max_chunks=1500
```
