# Experiments

Tracked, quantified history of every RAG configuration we try — so results are
real numbers, not "it felt better".

## How to run

```powershell
$py = "C:/Users/Administrator/miniconda3/envs/rag-tuning-lab/python.exe"
$env:PYTHONIOENCODING = "utf-8"

# Evaluate the current index + config, log the run
& $py -m rag_lab.experiment --config configs/diseases.yaml --label "baseline"

# Rebuild the index first (use when chunking/embedding/corpus changes)
& $py -m rag_lab.experiment --config configs/diseases.yaml --ingest --label "full-corpus"

# Ablation: same corpus, flip one knob — the leaderboard shows the delta
& $py -m rag_lab.experiment --config configs/diseases.yaml --ingest `
      --set chunking.prepend_title=false --label "no-prepend-title"
& $py -m rag_lab.experiment --config configs/diseases.yaml `
      --set rerank.mode=none --label "no-rerank"
```

## What gets recorded

- `runs.jsonl` — one JSON object per run (append-only, committed): timestamp,
  git sha, label, the knobs that matter (chunk size/overlap, prepend_title,
  hybrid weights, candidate_k, rerank mode, embedding model), corpus size,
  per-stage aggregate metrics, latency percentiles, and per-query detail.
- `LEADERBOARD.md` — regenerated table sorted by pipeline Recall@5.

## Metrics

Per query, against the expected disease id(s):

- **Recall@k / Hit@k** (k = 1, 3, 5, 10) — did the right doc make the top-k.
- **MRR** — 1 / rank of the first correct doc.
- **nDCG@k** — rank-weighted relevance.
- **latency** p50 / p90 / mean per query (ms).

Reported at four retrieval stages so we can see what each component buys:

| stage | what |
|---|---|
| `bm25_only` | keyword channel alone |
| `vector_only` | dense channel alone |
| `hybrid` | RRF fusion, no rerank |
| `hybrid_rerank` | fusion + cross-encoder (the configured pipeline) |

## Reference bands (industry rules of thumb, for sanity-checking our numbers)

| stage | typical Recall@5 |
|---|---|
| BM25 only | 50–75% |
| Vector only | 60–80% |
| Hybrid | 70–88% |
| Hybrid + Rerank | 78–92% |

Recall@10 is usually 5–10 points above Recall@5. These are bands to sanity-check
against, not targets to game — our eval set is small and will grow.
