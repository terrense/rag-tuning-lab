"""
================================================================================
compare_runs.py —— A/B 实验的"判决工具"：两个 run 的差异显著吗？
--------------------------------------------------------------------------------
LEADERBOARD 只能看均值，均值差 0.05 到底是提升还是噪声，要靠这里的
配对置换检验给 p 值。用法：

    python -m rag_lab.compare_runs --a tuned-defaults --b full-baseline
    python -m rag_lab.compare_runs --a A --b B --stage vector_only --metric mrr

约束：两个 run 必须在同一个评测集上（逐题 id 对齐做配对），且是新版
experiment.py 跑出来的（per_query 里带逐题 metrics）。同名 label 取最新一次。
================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_lab.metrics import bootstrap_ci, paired_permutation_test

RUNS_FILE = Path("experiments/runs.jsonl")


def _load_run(label: str) -> dict:
    """按 label 找最新一条 run 记录。"""
    match = None
    for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        if run.get("label") == label:
            match = run                      # 不 break：同名取最后（最新）一条
    if match is None:
        raise SystemExit(f"label '{label}' not found in {RUNS_FILE}")
    return match


def _per_query_values(run: dict, stage: str, metric: str) -> dict[str, float]:
    """run → {题id: 指标值}。老记录没存逐题指标时给出明确报错。"""
    out: dict[str, float] = {}
    for q in run.get("per_query", []):
        m = (q.get("metrics") or {}).get(stage)
        if m is None:
            raise SystemExit(
                f"run '{run['label']}' has no per-query metrics for stage '{stage}'.\n"
                "Re-run it with the current rag_lab.experiment (old logs lack them)."
            )
        if metric not in m:
            raise SystemExit(f"metric '{metric}' not found; available: {sorted(m)}")
        out[str(q.get("id"))] = float(m[metric] or 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired significance test between two runs.")
    ap.add_argument("--a", required=True, help="run label A（想证明更好的那个）")
    ap.add_argument("--b", required=True, help="run label B（基线）")
    ap.add_argument("--stage", default="hybrid_rerank",
                    choices=["bm25_only", "vector_only", "hybrid", "hybrid_rerank"])
    ap.add_argument("--metric", default="recall@5")
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    run_a, run_b = _load_run(args.a), _load_run(args.b)
    if run_a.get("eval_set") != run_b.get("eval_set"):
        raise SystemExit(f"eval sets differ ({run_a.get('eval_set')} vs {run_b.get('eval_set')}); "
                         "cross-eval-set comparison is meaningless.")

    va = _per_query_values(run_a, args.stage, args.metric)
    vb = _per_query_values(run_b, args.stage, args.metric)
    common = [qid for qid in va if qid in vb]      # 按题 id 对齐（顺序无关）
    if len(common) < len(va) or len(common) < len(vb):
        print(f"warning: only {len(common)} shared queries "
              f"(A has {len(va)}, B has {len(vb)}) — comparing the intersection.")
    if not common:
        raise SystemExit("no shared query ids between the two runs.")

    a_vals = [va[q] for q in common]
    b_vals = [vb[q] for q in common]
    mean_a, lo_a, hi_a = bootstrap_ci(a_vals)
    mean_b, lo_b, hi_b = bootstrap_ci(b_vals)
    test = paired_permutation_test(a_vals, b_vals, n_perm=args.n_perm)

    wins = sum(1 for x, y in zip(a_vals, b_vals) if x > y)
    losses = sum(1 for x, y in zip(a_vals, b_vals) if x < y)
    ties = len(common) - wins - losses

    print(f"\n{args.stage} / {args.metric}  (N={len(common)}, eval={run_a.get('eval_set')})")
    print(f"  A {args.a:30s} {mean_a:.3f}  95% CI [{lo_a:.3f}, {hi_a:.3f}]")
    print(f"  B {args.b:30s} {mean_b:.3f}  95% CI [{lo_b:.3f}, {hi_b:.3f}]")
    print(f"  mean diff (A-B) = {test['mean_diff']:+.3f}   win/tie/loss = {wins}/{ties}/{losses}")
    print(f"  paired permutation p = {test['p_value']:.4f}"
          + ("   → 显著 (p<0.05)" if test["p_value"] < 0.05 else "   → 不显著，差异可能是噪声"))
    # 逐题翻车/翻盘明细：看清 A 到底赢在哪些题、输在哪些题
    flips = [(q, va[q], vb[q]) for q in common if va[q] != vb[q]]
    if flips:
        print("  changed queries:")
        for q, x, y in sorted(flips, key=lambda t: t[1] - t[2]):
            print(f"    {'A>B' if x > y else 'A<B'}  {q:28s} A={x:.3f} B={y:.3f}")


if __name__ == "__main__":
    main()
