# Experiments

Tracked, quantified history of every RAG configuration we try — so results are
real numbers, not "it felt better".

Full experiment roadmap (P0 infra → P4 productionization): see [PLAN.md](PLAN.md).

## Statistical significance (read this before believing any delta)

A mean difference on a small eval set is usually noise. Two tools:

```powershell
# Is run A actually better than run B? Paired permutation test + bootstrap CIs
& $py -m rag_lab.compare_runs --a v2-bge-embedding --b v2-baseline
& $py -m rag_lab.compare_runs --a A --b B --stage vector_only --metric mrr
```

- `LEADERBOARD.md` now shows a bootstrap **95% CI** per run (per-query recall@5),
  plus an **eval** column — numbers from different eval sets are NOT comparable.
- Rule of thumb from our own data: at N=10, Recall@5=0.70 has CI [0.40, 1.00];
  at N=150 the CI narrows to roughly ±0.08. Hence eval set v2.

## Eval set v2 (`data/eval_queries_diseases_v2.yaml`)

119 queries = 10 hand-written anchors + 109 auto-generated:
department-stratified sampling over the 5942-disease corpus → deepseek-flash
writes questions in 4 rotating styles (symptom-led / name-led / exam-treatment /
colloquial) → rule checks (no disease-name leakage in symptom-led, dedup) →
deepseek-pro cross-model QC (rejected 41, e.g. non-specific symptom combos and
one corrupted source record). Rebuild: `python -m rag_lab.evalgen --n 150 --seed 42`.

## Generation-stage eval (`rag_lab.gen_eval`)

Retrieval metrics stop at "did the right chunk rank high"; this measures the
answer itself:

```powershell
& $py -m rag_lab.gen_eval --config configs/diseases.yaml --limit 20 --label gen-baseline
& $py -m rag_lab.gen_eval --config configs/diseases.yaml --set llm.roles.generate=deepseek-pro --label gen-pro
```

- Programmatic: citation_valid (no fabricated [n]), citation_precision /
  gold_cited (cites the right doc), abstain_correct (honest refusal when
  retrieval failed).
- LLM-as-judge: faithfulness / relevance 1-5, judged by `llm.roles.judge`
  (deepseek-pro) which must differ from the generator — no self-grading.
- Logged to `experiments/gen_runs.jsonl`.

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
