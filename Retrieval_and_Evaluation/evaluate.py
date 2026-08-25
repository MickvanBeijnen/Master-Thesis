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
"""

import math
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

TOKENIZER = "anserini"      # "anserini", "symbol" or "alphanum"

WORK_DIR   = Path("..")
RESULTS_DIR = WORK_DIR / f"results"
QRELS_PATH = WORK_DIR / f"qrels_{TOKENIZER}_test.tsv"

# Result files are auto-discovered by globbing results_test_*.tsv in RESULTS_DIR.
# No changes needed here when variants are added or removed.

CUTOFF = 10  # evaluate MRR@K and NDCG@K at this rank cutoff

# =============================================================================
# LOAD QRELS
# Qrels format: qid  query  docid  (no header, saved by bm25_baseline.py)
# We treat any entry as binary relevant (relevance = 1).
# =============================================================================

# Build dict of variant_name -> path from all result files found on disk
# RESULT_FILES = {
#     p.stem.replace("results_test_", ""): p
#     for p in sorted(RESULTS_DIR.glob("results_test_*.tsv"))
# }
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

def mrr_at_k(results_df: pd.DataFrame, qrels: set, k: int) -> float:
    """
    Mean Reciprocal Rank at cutoff k.
    For each query, find the rank of the first relevant document (1-indexed).
    Score = 1/rank if found within k, else 0. Average across all queries.
    Only queries that have at least one qrel entry are included.
    """
    scores = []
    for qid, group in results_df.groupby("qid"):
        if qid not in qids_with_qrels:
            continue
        top_k = group.sort_values("rank").head(k)
        rr = 0.0
        for _, row in top_k.iterrows():
            if (str(row["qid"]), str(row["docno"])) in qrels:
                rr = 1.0 / row["rank"]
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(results_df: pd.DataFrame, qrels: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at cutoff k.
    Binary relevance: gain = 1 if relevant, 0 otherwise.
    DCG  = sum(rel_i / log2(i+1)) for i in 1..k
    IDCG = sum(1 / log2(i+1)) for i in 1..min(num_relevant, k)
    Only queries that have at least one qrel entry are included.
    """
    scores = []
    for qid, group in results_df.groupby("qid"):
        if qid not in qids_with_qrels:
            continue

        top_k = group.sort_values("rank").head(k)

        # DCG
        dcg = 0.0
        for _, row in top_k.iterrows():
            if (str(row["qid"]), str(row["docno"])) in qrels:
                dcg += 1.0 / math.log2(row["rank"] + 1)

        # IDCG — ideal ranking: all relevant docs at the top
        num_relevant = sum(
            1 for docid in qrels_df[qrels_df["qid"] == qid]["docid"]
            if (qid, str(docid)) in qrels
        )
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(num_relevant, k)))

        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0

# =============================================================================
# EVALUATE ALL VARIANTS
# =============================================================================

results_summary = []

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

    mrr  = mrr_at_k(df,  relevant, CUTOFF)
    ndcg = ndcg_at_k(df, relevant, CUTOFF)

    print(f"[{variant}] MRR@{CUTOFF}  = {mrr:.4f}")
    print(f"[{variant}] NDCG@{CUTOFF} = {ndcg:.4f}\n")

    results_summary.append({
        "variant":          variant,
        f"MRR@{CUTOFF}":    round(mrr,  4),
        f"NDCG@{CUTOFF}":   round(ndcg, 4),
        "queries_evaluated": num_queries,
    })

# =============================================================================
# SUMMARY TABLE
# =============================================================================

summary_df = pd.DataFrame(results_summary)
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

summary_df.to_csv(WORK_DIR / "evaluation_results.tsv", sep="\t", index=False)
print(f"\nSaved to {WORK_DIR}/evaluation_results.tsv")
