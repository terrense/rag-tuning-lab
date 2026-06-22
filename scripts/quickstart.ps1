param(
    [string]$Config = "configs/chroma.yaml"
)

$ErrorActionPreference = "Stop"

python -m rag_lab.ingest --config $Config
python -m rag_lab.query --config $Config --query-id q_rerank
python -m rag_lab.sweep --config $Config --query-id q_rerank --vary retrieval.candidate_k=6,12 --vary rerank.mode=none,bm25
