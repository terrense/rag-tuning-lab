#!/usr/bin/env sh
# GPU 实验链。三条硬教训固化在这里：
#   1) set -e：任何一步失败立即停，绝不继续跑后面步骤（之前 |tail 吞退出码 → 孤儿进程）
#   2) 不用 |tail：完整输出进日志，退出码真实传播
#   3) 每个 embedding 模型用独立 chroma_dir：彻底避免共享 SQLite 被半成品 collection
#      的 delete_collection 卡死（bge-m3 之前反复卡在删残缺 HNSW 索引这一行）
set -e
PY="C:/Users/Administrator/miniconda3/envs/rag-tuning-lab/python.exe"
export HF_HOME=F:/hf_cache
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."

V2="--set paths.eval_queries=data/eval_queries_diseases_v2.yaml"
W="--set retrieval.bm25_weight=0.3 --set retrieval.vector_weight=0.7"
# bge-small 用它自己已建好的库（在共享目录里，已完好 count=27468），只做 E2 精排评测
BGE="--set embedding.model=BAAI/bge-small-zh-v1.5 --set vector_store.collection=rag_lab_diseases_bge --set paths.chunks_cache=storage/chunks_diseases_bge.jsonl"
# bge-m3 用全新独立目录，reset_on_ingest 在新目录里无旧 collection 可删 → 不会卡
M3="--set embedding.model=BAAI/bge-m3 --set vector_store.collection=bgem3 --set paths.chroma_dir=storage/chroma_bgem3 --set paths.chunks_cache=storage/chunks_diseases_bgem3.jsonl"

"$PY" -c "import torch;assert torch.cuda.is_available();print('[chain] GPU:',torch.cuda.get_device_name(0))"

echo "=== STEP1 E1-arm3: bge-m3 full ingest + eval (fresh chroma dir)"
"$PY" -X utf8 -u -m rag_lab.experiment --config configs/diseases.yaml --ingest \
  --set source.structured_max_records=0 $M3 $V2 $W --label v2-bgem3-w0.3
echo "STEP1_DONE"

echo "=== STEP2 E2: bge-reranker-v2-m3 on bge-small index"
"$PY" -X utf8 -u -m rag_lab.experiment --config configs/diseases.yaml \
  $BGE $V2 $W --set rerank.model=BAAI/bge-reranker-v2-m3 --label v2-bge-w0.3-rerankv2
echo "STEP2_DONE"

echo "ALL_GPU_CHAIN_DONE"
