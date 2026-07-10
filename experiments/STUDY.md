# RAG 面试速查 · STUDY.md

> 本项目每个能力的：**核心概念 → 我们的数据 → 可能被追问的问题 + 答法**。
> 配套"亲手跑一遍"命令在最后一节。原则：**只讲有数据支撑的，诚实说局限。**

---

## 0. 一句话故事线（面试开场自我介绍用）

> "我做了个医疗 RAG 实验台，锚定真实的 5942 条疾病语料。**检索**上做了 embedding 和
> reranker 的显著性对比（还用数据否决了不划算的大模型）；**评测**上从只看检索扩展到
> 生成端的 faithfulness 和引用准确率，评测集带统计检验；**多模态**做了表格/OCR 的四臂
> 基准，证明'字符识别对≠字段结构对'；**生产化**做了带分阶段 tracing 和语义缓存的
> FastAPI 服务；**进阶技法**做了 CRAG、parent-document、contextual retrieval，并诚实地
> 量出它们在什么语料上才值得用。"

---

## 1. 统计严谨性（贯穿一切的底座）

**概念**：小评测集上的均值差多半是噪声。两个工具：
- **bootstrap 置信区间**：把 N 道题当样本，有放回重抽 N 道算均值，重复上万次，取 2.5%~97.5% 分位。区间宽度直观反映"评测集有多小"。
- **配对置换检验（paired permutation test）**：同一批题上 A、B 的逐题差值，随机翻符号上万次，看真实差值有多极端 → p 值。

**我们的数据**：N=10 时 Recall@5=0.70 的 CI 是 `[0.40, 1.00]`（宽 0.60）；N=119 收窄到 ±0.08。旧排行榜上 0.70 vs 0.60 的"提升"，配对检验 p≈0.25——**是噪声**。

**追问 & 答法**：
- Q：为什么不用 t 检验？
  A：检索指标是 0/1 命中、截断的 MRR，根本不是正态分布，t 检验前提不成立，所以用无分布假设的重采样。
- Q：为什么用"配对"？
  A：同一道题两边都答了，比较逐题差值能消掉题目难度本身的方差，比当独立样本灵敏得多。

---

## 2. 检索：embedding 与 reranker（都带 p 值）

**概念**：两阶段检索漏斗——召回求全（便宜、粗：向量+BM25+RRF），精排求准（贵、精：cross-encoder）。

**我们的数据（v2, N=119）**：

| 变量 | 结论 | 数字 |
|---|---|---|
| 英文 MiniLM → 中文 bge-small-zh | ✅ 决定性 | 向量召回 R@5 0.32→0.78，p=0.0001 |
| bge-small → 更大的 bge-m3 | ❌ 不值得 | 0.78→0.80，p=0.69 **不显著** |
| 换 reranker → bge-reranker-v2-m3 | ✅ 显著 | MRR 0.66→0.72，p=0.0022；R@1 0.58→0.66 |

**两个反直觉洞见**：
1. **"更大更新"不一定值得**：bge-m3 是 20 倍大的模型、要占 GPU，但多花的算力没换来可测提升 → 小模型是对的选择。
2. **精排的价值在"排序"不在"召回"**：换强 reranker 后 Recall@5 几乎不动（对的文档本就在前5），但 R@1 大涨（顶到第1位）。**指标要看对**。

**追问 & 答法**：
- Q：为什么中文库里 BM25 权重反而调高（0.3→0.6）？
  A：多语言 MiniLM 对中文医疗文本弱（向量召回仅 0.20），所以 BM25 是更强的通道，反常识地加重它。换了中文 embedding 后向量变强，最优 BM25 权重就回落到 0.3——**换 embedding 会引发全局重调参**。
- Q：RRF 为什么不直接加权分数？
  A：向量的余弦分和 BM25 分量纲不同没法相加；RRF 只看排名，两路都靠前的文档融合分自然最高。

---

## 3. 生成端评测（检索指标好 ≠ 答案好）

**概念**：检索指标只管"对的文档有没有排上来"，答案本身的质量要单独评。两类：
- **程序可算**：引用合法率（有没有编造资料号）、引用精确率、诚实弃答率。
- **LLM-as-judge**：faithfulness（每个论断有没有资料支撑）、relevance，1-5 分。

**关键原则**：**裁判必须 ≠ 生成模型**（自己评自己有自偏袒）。我们生成用 MiniMax、裁判用 deepseek-pro。

**追问 & 答法**：
- Q：LLM 当裁判不可靠怎么办？
  A：①裁判≠被评模型；②答案顺序随机化去位置偏差；③硬指标（引用精确率）程序算不受裁判影响，和主观分交叉验证；④裁判解析失败算"测量失败"剔除，不当质量0分。
- Q：没有 ground truth 怎么评？（我们评测集有金标，但生产常没有）
  A：faithfulness 是无需金标的——只要求答案不超出检索到的资料，是"接地性"而非"正确性"。

---

## 4. 多模态：表格 / OCR 四臂基准

**概念**：表格是 RAG 最易翻车处。五个经典坑：合并单元格、跨页无表头、多级表头、数值单位分离、列错位。**评估口径**：不报笼统字符准确率，报**字段级 exact match** + **行级完整率**，分数字件/扫描件两档。

**我们的数据（扫描件，行级准确率）**：

| 坑 | 裸OCR(RapidOCR) | PP-Structure | GOT-OCR2(VLM) |
|---|:---:|:---:|:---:|
| 合并单元格 | 0.00 | 0.75 | 0.56 |
| 多级表头 | 0.00 | 0.80 | 0.80 |
| 列错位 | 0.00 | 0.50 | **1.00** |

所有臂**字符命中都 0.86-1.00**，差距全在结构还原。

**追问 & 答法**：
- Q：OCR 准确率能到多少？
  A：**不能笼统说一个数字**。分三层看：字符级（清晰打印体高）、字段级（检查项目/结果/单位/参考范围是否都对）、表格行级（整行含数值单位是否错位）。我不会说"99%"，而是说"关键字段抽取在清晰件上 90%+，复杂/跨页/低质件明显下降，所以加了字段校验和低置信降级"。
- Q：VLM（GOT-OCR2）vs 结构化管线（PP-Structure）怎么选？
  A：要稳定结构用 PP-Structure；要抗版面畸变用 VLM（列错位它满分）。但 VLM 会幻觉结构（把提示列误读成合并），生产上可 VLM 出结构 + 规则校验兜底。

---

## 5. 生产服务层（Track A）

**概念**：从 CLI 一次性进程 → 常驻 HTTP 服务。三块：
- **分阶段 tracing**：每请求记 embed/retrieve/rerank/generate 各阶段延迟 + token + 成本 → 分布式追踪的最小内核。
- **语义缓存**：query 向量余弦匹配，相似就复用答案（不是精确字符匹配）。
- **SSE 流式**：先发引用、再逐 token，改善首 token 体验。

**我们的数据**：语义缓存原样重复命中 **15ms vs 4000ms，快 ~270 倍**。

**追问 & 答法**：
- Q：语义缓存阈值怎么定？
  A：实测出来的——bge-small 下轻度改写相似度 ~0.99（该命中）、无关问题 ~0.25（该隔离）、重度长句改写只有 ~0.67（命中不了）。定 0.90 保守优先：**宁可不命中，不可张冠李戴把别人的答案返给用户**。
- Q：缓存怎么失效？
  A：语料更新时清缓存 + TTL；只缓存成功生成、非拒答的结果。

---

## 6. 进阶技法（Track B）— 重点讲"何时该用"

**CRAG（检索自纠错）**：检索完自评质量 → proceed / 改写重检 / 诚实拒答。
- 数据：真失败时正确拒答 66%，但便宜评估器对症状题太严 → **25% 误拒答**。
- 讲法：这是"**拒答的精确率/召回率权衡**"。想少幻觉就会多误拒。杠杆：换强评估器 或 降拒答阈值。医疗场景偏保守（可接受高拒答）。

**Parent-Document（小块检索、大块喂）**：检索用小块（准），喂 LLM 用整篇父文档（全）。
- 数据：faithfulness 4.20→4.25（CI 重叠，**无显著增益**）。

**Contextual Retrieval（Anthropic）**：建库时给每块加一句 LLM 生成的上下文前缀。
- 机制验证：只有"【症状】"的孤块获得了"介绍急性呼吸窘迫综合征的症状/治疗/检查"前缀。

**贯穿三者最重要的一课（面试高光）**：
> 这三个技法在我们语料上**都几乎无增益**——不是技法没用，是**这个语料的块本来就不
> 丢上下文**（疾病记录小 + 每块都 prepend 病名）。它们的价值在"长文档被切散、块失去
> 兄弟上下文"的场景。**知道一个技法什么时候是过度设计，比会实现它更值钱。**

---

## 7. 工程血泪（面试聊落地时的弹药）

- **`命令 | tail` 吞退出码**：shell 拿到 tail 的成功码，失败链继续跑 → 孤儿进程 + 半成品数据库 → 反复卡死。修复：`set -e`、不用 `|tail`、每模型独立 chroma 目录。用 **py-spy** 抓到卡在 `delete_collection`。
- **flash 是推理模型**：max_tokens 给太少（80）会被 `<think>` 吃光、剥离后为空。判官/生成给足 token + 解析失败重试。
- **透明加密软件**：浏览器进程写的 PDF 变密文 → 改用 reportlab 纯 Python 生成。
- **Blackwell 5070 显卡**：需要 cu128 版 torch；`pip install torch` 会因"已满足"跳过，得 `--force-reinstall`。
- **裁判要看对上下文**：parent 模式喂父文档、裁判却看 chunk → 不公平压低分。

---

## 8. 亲手跑一遍（直观感受）

**⚠ 每次开新的 PowerShell 窗口，先跑这三行**（关键：项目装在 `rag-tuning-lab`
环境里，不是 `base`；不激活就会报 `No module named 'rag_lab'`）：
```powershell
conda activate rag-tuning-lab        # ← 最容易忘的一步！base 环境里跑不了
$env:PYTHONIOENCODING = "utf-8"       # 中文不乱码
$env:HF_HOME = "F:/hf_cache"          # 模型缓存在 F 盘
cd F:\RAG_experiment
```
> 激活后命令行前缀会从 `(base)` 变成 `(rag-tuning-lab)`，之后直接用 `python`。

### A. 感受检索（最快，无需 LLM）
```powershell
python -m rag_lab.query --config configs/diseases.yaml --query "大脚趾夜里剧痛红肿尿酸高"
# 看它把"痛风/高尿酸血症"排上来；改 --query 试各种症状描述
```

### B. 感受带引用的生成
```powershell
python -m rag_lab.ask --config configs/diseases.yaml --query "苯中毒的症状和检查"
# 答案后带 [1][2] 引用编号，可溯源
```

### C. 感受统计显著性（本项目的灵魂）
```powershell
python -m rag_lab.compare_runs --a v2-bge-w0.3-rerankv2 --b v2-baseline --stage hybrid_rerank --metric mrr
# 看它输出 mean diff + p 值 + 逐题胜负。p<0.05 才算真提升
type experiments\LEADERBOARD.md    # 排行榜，每行带 95% CI
```

### D. 感受生产服务层（开两个终端）
```powershell
# 终端1（先激活 rag-tuning-lab 环境）：起服务，模型常驻，启动约 30s
python -m rag_lab.serve --config configs/diseases.yaml --port 8000

# 然后浏览器直接开 http://127.0.0.1:8000/docs —— FastAPI 自带交互式文档，
# 点 /ask 的 "Try it out" 输入 {"query":"哮喘怎么检查"} 就能看 trace/cache，最直观。

# 或终端2（也要先 conda activate rag-tuning-lab）用命令行打：
python -c "import requests; print(requests.post('http://127.0.0.1:8000/ask', json={'query':'哮喘怎么检查'}).json()['trace'])"
python -c "import requests; print(requests.get('http://127.0.0.1:8000/metrics').json())"   # 同问第二次看 cache 命中
```

### E. 感受 CRAG 诚实拒答
```powershell
python -m rag_lab.crag --config configs/diseases.yaml --set source.structured_max_records=0 --query "如何用Python写快速排序"
# 语料里没有 → decision=abstain，诚实说"无法确定"，不硬编
python -m rag_lab.crag --config configs/diseases.yaml --set source.structured_max_records=0 --query "痛风怎么治"
# 有 → decision=proceed
```

### F. 感受表格/OCR 基准
```powershell
python scripts/table_eval.py     # 数字PDF：naive vs robust，看修复把 t1 0.19→1.00
# OCR 臂要换 ocr-lab 环境：
conda activate ocr-lab
python scripts/ocr_table_eval.py --engine rapidocr   # 看 text_hit 高但 field_em 低
conda activate rag-tuning-lab    # 跑完记得切回来
```

> 建议顺序：先 A/B（看检索和生成），再 C（理解为什么要显著性），再 D（生产服务），
> 最后 E/F（进阶）。每一步的详细文档在 `experiments/` 对应的 .md 里。
>
> **踩坑速查**：报 `No module named 'rag_lab'` = 没激活 rag-tuning-lab 环境；
> 报 `py 无法识别` = 别输 `py`，本文档已改用 `python`；`$py 无效对象` = 那是旧写法，
> 现在不用变量了，直接 `python`。
