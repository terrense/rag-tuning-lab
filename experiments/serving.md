# 服务层与可观测性（Track A）

把实验台从"CLI 一次性进程"升级成"能上线的常驻服务"——"生产级"叙事最缺的一块。

## 启动

```powershell
$py = "C:/Users/Administrator/miniconda3/envs/rag-tuning-lab/python.exe"
$env:HF_HOME = "F:/hf_cache"
& $py -m rag_lab.serve --config configs/diseases.yaml --port 8000
```

启动时预热 embedder + 检索资产（chunks/BM25）+ 语义缓存 → 请求不再冷启动。

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活 + 预热状态 |
| GET | `/metrics` | 请求数、延迟 p50/p95/p99、缓存命中率 |
| POST | `/search` | 只检索（最快，无 LLM），返回命中 + trace |
| POST | `/ask` | 检索 + 带引用生成，走语义缓存，返回答案 + sources + trace |
| POST | `/ask/stream` | 同上，SSE 流式（先发 sources，再逐 token，末尾 trace） |

请求体：`{"query": "...", "history": [...], "top_k": 5, "no_cache": false, "no_llm": false}`

## 三块能力（都实测通过）

### 1. 分阶段可观测（`trace.py`）
每个请求一条 Trace，`/ask` 响应里带：
```json
"trace": {
  "spans": [{"name":"retrieve","ms":360}, {"name":"generate","ms":4029}],
  "total_ms": 4389, "cost_usd": 0.000634,
  "usage": {"MiniMax-M3": {"prompt":1116, "completion":249}}
}
```
落盘 `logs/traces.jsonl`。token 用量由 `llm.chat(trace=...)` 无侵入记账，成本按
`trace.PRICING` 价目表折算（估算，可改）。**"哪一步慢、哪一步烧钱"一目了然**：
retrieve 通常几百 ms，generate 是大头（3-8s）。

### 2. 语义缓存（`semantic_cache.py`）
query 向量余弦匹配，超阈值直接返回上次答案。实测：
- **原样重复问：命中，15ms vs 4000ms 生成 —— 快 ~270 倍。**
- 阈值调参有真实依据：bge-small 下轻度改写（"糖尿病怎么治疗"↔"如何医治"≈0.99）
  能命中，无关问题（≈0.25）安全隔离；**重度长句改写只有 ≈0.67，命中不了**——
  这是个诚实的局限：要接住重度改写得上更强 embedding，或先对 query 做规范化改写。
- 阈值默认 0.90（保守优先——宁可不命中，不可张冠李戴把别人的答案返给用户）。
- 只缓存"成功生成且非拒答"的结果。

### 3. SSE 流式（`/ask/stream`）
先发 `sources` 事件，再逐 `token` 事件流式吐答案，末尾 `trace`。体验上首 token
更快。推理模型的 `<think>` 段用状态机在流里剥离。

## 待办（生产化继续）
- 并发压测（locust）拿真实 QPS / p99；trace 已就绪，压测数据可直接画延迟瀑布。
- 缓存失效策略（TTL、语料更新时清缓存）。
- 鉴权 / 限流 / 请求体校验加固。
- docker-compose 一键起（含 Milvus infra）。
