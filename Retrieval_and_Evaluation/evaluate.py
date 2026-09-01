"""
evaluate.py

Computes MRR@10 and NDCG@10 for all retrieval result files produced by
bm25_baseline.py, using the ORCAS qrels as relevance judgements.

MRR@10  — Mean Reciprocal Rank at cutoff 10. The standard MS MARCO metric.
           Focuses on where the first relevant document appears.
NDCG@10 — Normalized Discounted Cumulative Gain at cutoff 10.
           Rewards putting more relevant documents higher in the ranking.

For ORCAS (binary click relevance, typically one relevant doc per query),
MRR@10 and NDCG@10 will be similar. Both are reported for completeness.

Statistical significance is assessed using the Wilcoxon signed-rank test
(per-query scores, two-tailed) against the baseline variant. Bonferroni
correction is applied to account for multiple comparisons.
"""

import math
import pandas as pd
from pathlib import Path
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

TOKENIZER = "anserini"      # "anserini", "symbol" or "alphanum"

WORK_DIR    = Path("..")
RESULTS_DIR = WORK_DIR / f"results"
QRELS_PATH  = WORK_DIR / f"qrels_{TOKENIZER}_test.tsv"

# Name of the baseline variant to compare all others against.
# Must match the variant name auto-discovered from result files.
SIGNIFICANCE_BASELINE = f"combined_{TOKENIZER}_maxvalrlm10_maxvalcg5_a1_00_b0_00"   # For interpolation experiment
# SIGNIFICANCE_BASELINE = f"cg_anserini_maxval5_alpha_0_00"     # For Doc Aug Prototype

# Significance level before Bonferroni correction
ALPHA = 0.05

# Result files are auto-discovered by globbing results_test_*.tsv in RESULTS_DIR.
# No changes needed here when variants are added or removed.

CUTOFF = 10  # evaluate MRR@K and NDCG@K at this rank cutoff

# =============================================================================
# LOAD QRELS
# Qrels format: qid  query  docid  (no header, saved by bm25_baseline.py)
# We treat any entry as binary relevant (relevance = 1).
# =============================================================================

RESULT_FILES = {
    p.stem.replace("results_test_", ""): p
    for p in sorted(RESULTS_DIR.glob("results_test_*.tsv"))
    if TOKENIZER in p.stem.replace("results_test_", "").split("_")
}

if not RESULT_FILES:
    print(f"No result files found in {RESULTS_DIR}. Run bm25_baseline.py first.")
    exit(1)

print(f"Auto-discovered {len(RESULT_FILES)} result file(s):")
for name, path in RESULT_FILES.items():
    print(f"  {name} -> {path}")
print()

print(f"Loading qrels from {QRELS_PATH} ...")
qrels_df = pd.read_csv(
    QRELS_PATH, sep="\t", header=None,
    names=["qid", "query", "docid"], dtype=str
)

# Build a set of (qid, docid) relevant pairs for fast lookup
relevant = set(zip(qrels_df["qid"].str.strip(), qrels_df["docid"].str.strip()))
qids_with_qrels = set(qrels_df["qid"].str.strip().unique())
print(f"Loaded {len(relevant):,} relevant pairs across {len(qids_with_qrels):,} queries.\n")

# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def mrr_at_k(results_df: pd.DataFrame, qrels: set, k: int) -> tuple[float, dict]:
    """
    Mean Reciprocal Rank at cutoff k.
    Returns (mean_score, per_query_scores dict keyed by qid).
    """
    per_query = {}
    for qid, group in results_df.groupby("qid"):
        if qid not in qids_with_qrels:
            continue
        top_k = group.sort_values("rank").head(k)
        rr = 0.0
        for _, row in top_k.iterrows():
            if (str(row["qid"]), str(row["docno"])) in qrels:
                rr = 1.0 / row["rank"]
                break
        per_query[qid] = rr
    mean = sum(per_query.values()) / len(per_query) if per_query else 0.0
    return mean, per_query


def ndcg_at_k(results_df: pd.DataFrame, qrels: set, k: int) -> tuple[float, dict]:
    """
    Normalized Discounted Cumulative Gain at cutoff k.
    Returns (mean_score, per_query_scores dict keyed by qid).
    """
    per_query = {}
    for qid, group in results_df.groupby("qid"):
        if qid not in qids_with_qrels:
            continue

        top_k = group.sort_values("rank").head(k)

        dcg = 0.0
        for _, row in top_k.iterrows():
            if (str(row["qid"]), str(row["docno"])) in qrels:
                dcg += 1.0 / math.log2(row["rank"] + 1)

        num_relevant = sum(
            1 for docid in qrels_df[qrels_df["qid"] == qid]["docid"]
            if (qid, str(docid)) in qrels
        )
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(num_relevant, k)))

        per_query[qid] = dcg / idcg if idcg > 0 else 0.0

    mean = sum(per_query.values()) / len(per_query) if per_query else 0.0
    return mean, per_query


# =============================================================================
# EVALUATE ALL VARIANTS — collect per-query scores for significance testing
# =============================================================================

results_summary = []
per_query_mrr  = {}   # variant -> {qid -> score}
per_query_ndcg = {}   # variant -> {qid -> score}

for variant, path in RESULT_FILES.items():
    if not path.exists():
        print(f"[{variant}] Skipped — file not found: {path}")
        continue

    print(f"[{variant}] Loading {path} ...")
    df = pd.read_csv(path, sep="\t", dtype={"qid": str, "docno": str, "rank": int})
    df["qid"]   = df["qid"].str.strip()
    df["docno"] = df["docno"].str.strip()

    num_queries = df["qid"].nunique()
    print(f"[{variant}] {len(df):,} result rows across {num_queries:,} queries.")

    mrr,  pq_mrr  = mrr_at_k(df,  relevant, CUTOFF)
    ndcg, pq_ndcg = ndcg_at_k(df, relevant, CUTOFF)

    per_query_mrr[variant]  = pq_mrr
    per_query_ndcg[variant] = pq_ndcg

    print(f"[{variant}] MRR@{CUTOFF}  = {mrr:.4f}")
    print(f"[{variant}] NDCG@{CUTOFF} = {ndcg:.4f}\n")

    results_summary.append({
        "variant":            variant,
        f"MRR@{CUTOFF}":      round(mrr,  4),
        f"NDCG@{CUTOFF}":     round(ndcg, 4),
        "queries_evaluated":  num_queries,
    })

# =============================================================================
# SIGNIFICANCE TESTING
# Wilcoxon signed-rank test (two-tailed) vs baseline, with Bonferroni correction
# =============================================================================

if SIGNIFICANCE_BASELINE in per_query_mrr:
    non_baseline = [v for v in per_query_mrr if v != SIGNIFICANCE_BASELINE]
    n_comparisons = len(non_baseline)
    corrected_alpha = ALPHA / n_comparisons if n_comparisons > 0 else ALPHA

    print("=" * 60)
    print(f"SIGNIFICANCE TESTS vs '{SIGNIFICANCE_BASELINE}'")
    print(f"Wilcoxon signed-rank test, two-tailed")
    print(f"Bonferroni correction: alpha {ALPHA} / {n_comparisons} = {corrected_alpha:.6f}")
    print("=" * 60)

    baseline_mrr_scores  = per_query_mrr[SIGNIFICANCE_BASELINE]
    baseline_ndcg_scores = per_query_ndcg[SIGNIFICANCE_BASELINE]

    sig_results = []
    for variant in non_baseline:
        # Align on shared query IDs
        shared_qids = sorted(
            set(baseline_mrr_scores) & set(per_query_mrr[variant])
        )
        if len(shared_qids) < 10:
            print(f"[{variant}] Skipped — too few shared queries ({len(shared_qids)})")
            continue

        b_mrr  = [baseline_mrr_scores[q]         for q in shared_qids]
        v_mrr  = [per_query_mrr[variant][q]       for q in shared_qids]
        b_ndcg = [baseline_ndcg_scores[q]         for q in shared_qids]
        v_ndcg = [per_query_ndcg[variant][q]      for q in shared_qids]

        _, p_mrr  = stats.wilcoxon(b_mrr,  v_mrr,  zero_method="wilcox", alternative="two-sided")
        _, p_ndcg = stats.wilcoxon(b_ndcg, v_ndcg, zero_method="wilcox", alternative="two-sided")

        sig_mrr  = "✓" if p_mrr  < corrected_alpha else " "
        sig_ndcg = "✓" if p_ndcg < corrected_alpha else " "

        print(f"[{variant}]")
        print(f"  MRR@{CUTOFF}:  p={p_mrr:.6f}  [{sig_mrr}] {'significant' if sig_mrr == '✓' else 'not significant'}")
        print(f"  NDCG@{CUTOFF}: p={p_ndcg:.6f}  [{sig_ndcg}] {'significant' if sig_ndcg == '✓' else 'not significant'}")

        sig_results.append({
            "variant":          variant,
            "n_queries":        len(shared_qids),
            f"p_MRR@{CUTOFF}":  round(p_mrr,  6),
            f"p_NDCG@{CUTOFF}": round(p_ndcg, 6),
            f"sig_MRR@{CUTOFF}":  p_mrr  < corrected_alpha,
            f"sig_NDCG@{CUTOFF}": p_ndcg < corrected_alpha,
        })

    if sig_results:
        sig_df = pd.DataFrame(sig_results)
        sig_df.to_csv(WORK_DIR / "significance_results.tsv", sep="\t", index=False)
        print(f"\nSaved significance results to {WORK_DIR}/significance_results.tsv")

else:
    print(f"WARNING: Baseline variant '{SIGNIFICANCE_BASELINE}' not found — significance tests skipped.")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

summary_df = pd.DataFrame(results_summary)
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

summary_df.to_csv(WORK_DIR / "evaluation_results.tsv", sep="\t", index=False)
print(f"\nSaved to {WORK_DIR}/evaluation_results.tsv")
