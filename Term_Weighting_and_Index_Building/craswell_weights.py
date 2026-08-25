"""
craswell_weights.py
─────────────────────────────
Generates per-(doc, term) weights via Personalized PageRank (PPR) on the
ORCAS click graph, in the same TSV format expected by sort_weights.py and
ReweightIndex.java:

    traversal_idx \t doc_id \t term \t probability

All configuration is in the CONFIGURATION section below — edit before running.

Two modes, selected by the EXPAND flag:

  EXPAND = False  (default)
    Aggregate PPR mode. Runs a single PPR power iteration seeded at ALL train
    queries simultaneously (exact by linearity of PPR), giving each document a
    scalar relevance score. That score is distributed uniformly (or by TF if
    WEIGHT_BY_TF = True) across the document's own terms. Suitable for
    cg / cg_multi in ReweightIndex.java.

  EXPAND = True
    Document expansion mode. Applies Craswell et al.'s two-stage graph pruning,
    then computes co-click similarity between documents via a query-pivot
    approach. For each document, takes the top TOP_K most similar neighbors and
    marginalizes their term distributions using RLM Method 2 (Lavrenko & Croft
    2001). New terms (not in the original document) are included in the output
    so that ReweightIndex.java's cg_expand mode can inject them. Suitable for
    cg_expand in ReweightIndex.java.

Output is NOT yet sorted; pipe through sort_weights.py afterwards.

Design notes
────────────
• Train/test leakage: ORCAS clicks are filtered to query_ids present in
  QRELS_TRAIN_PATH before the graph is built, mirroring the split used for
  RLM. Test queries never enter the click graph.
• Aggregate PPR: by linearity of the PPR recurrence, Σ_q π_q equals the fixed
  point of one power iteration with an un-normalized restart vector (one entry
  per train query). Exact, not an approximation — see ppr_aggregate().
• Expansion: query-pivot co-click similarity replaces per-doc PPR walks.
  sim(d,d′) = Σ_q w(q) where the sum is over queries clicked by both d and d′,
  and w(q) = 1/degree(q). O(32M ops) versus O(724 hours) for sequential walks.
• traversal_idx is left as -1 in the raw output; sort_weights.py fills it in.
"""

import gzip
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from codecarbon import OfflineEmissionsTracker
from tqdm import tqdm

# =============================================================================
# CONFIGURATION — edit before running / submitting
# =============================================================================

TOKENIZER = "anserini"   # "anserini", "symbol" or "alphanum"

WORK_DIR = Path("..")

ORCAS_PATH       = WORK_DIR / f"orcas_preprocessed_{TOKENIZER}.tsv"
QRELS_TRAIN_PATH = WORK_DIR / f"qrels_{TOKENIZER}_train.tsv"
QRELS_TEST_PATH  = WORK_DIR / f"qrels_{TOKENIZER}_test.tsv"
CORPUS_PATH      = WORK_DIR / f"corpus_{TOKENIZER}" / "msorcas.jsonl"
INDEX_DIR        = WORK_DIR / f"anserini_index_{TOKENIZER}"

# --- Output ---
# Raw (unsorted) output — pipe through sort_weights.py afterwards.
# EXPAND = False → cg weights (for cg / cg_multi)
# EXPAND = True  → expansion weights (for cg_expand)
EXPAND = True

OUTPUT_FILE = WORK_DIR / (
    f"clickgraph_expand_weights_{TOKENIZER}_raw.tsv"
    if EXPAND else
    f"clickgraph_weights_{TOKENIZER}_raw.tsv"
)

# --- PPR hyperparameters (aggregate mode) ---
PPR_ALPHA = 0.15   # restart probability
PPR_ITERS = 20     # power-iteration steps; 20 is typically well-converged

# --- Term weight distribution (aggregate mode only) ---
WEIGHT_BY_TF = True   # True → distribute doc score proportional to TF
                        # False → uniform over unique terms

# --- Expansion hyperparameters (expand mode only) ---
TOP_K = 10   # number of neighbor documents to marginalize over (RLM feedback set size)

# --- Smoke-test limit ---
# Set to an integer (e.g. 50_000) to stop reading ORCAS early for quick tests.
# Set to None for a full run.
MAX_CLICKS = None

# --- CodeCarbon ---
# Uses OfflineEmissionsTracker (no internet required on the cluster).
# Emissions CSV is written to WORK_DIR; a new row is appended on each run so
# the full energy history across multiple Slurm jobs is preserved.
# NLD = Netherlands (ISO 3166-1 alpha-3 code for your cluster location).
COUNTRY_ISO_CODE = "NLD"

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# TOKENISER
# =============================================================================

def tokenize(text: str) -> list[str]:
    """Whitespace tokeniser matching Anserini's default behaviour."""
    tokens = text.lower().split()
    if TOKENIZER == "nonum":
        tokens = [t for t in tokens if not t.isdigit()]
    return tokens

# =============================================================================
# DATA LOADING
# =============================================================================

def load_train_qids() -> set[str]:
    """Load training query IDs from qrels TSV (qid, query_text, doc_id)."""
    qids: set[str] = set()
    with open(QRELS_TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                qids.add(parts[0].strip())
    log.info("Loaded %d unique train qids from %s", len(qids), QRELS_TRAIN_PATH)
    return qids


def build_click_graph(train_qids: set[str]):
    """
    Parse ORCAS TSV and return COO triplets for the bipartite click graph.
    Filters to train_qids to prevent test-query leakage.
    Supports 3-column (qid, query_text, doc_id) and 4-column
    (qid, query_text, doc_id, url) ORCAS formats.
    """
    log.info("Reading ORCAS click data from %s", ORCAS_PATH)
    log.info("Filtering to %d train qids (test queries excluded)", len(train_qids))

    click_counts: dict[tuple[str, str], int] = defaultdict(int)
    n_seen = 0
    n_kept = 0

    opener = gzip.open if str(ORCAS_PATH).endswith(".gz") else open
    with opener(ORCAS_PATH, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if MAX_CLICKS is not None and i >= MAX_CLICKS:
                log.info("Stopped reading ORCAS at %d lines (MAX_CLICKS)", MAX_CLICKS)
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                qid, query_text, doc_id, _ = parts
            elif len(parts) == 3:
                qid, query_text, doc_id = parts
            else:
                raise ValueError(
                    f"ORCAS line has no query_id column, cannot apply "
                    f"train/test filter safely: {parts!r}"
                )
            n_seen += 1
            if qid.strip() not in train_qids:
                continue
            n_kept += 1
            click_counts[(query_text.strip(), doc_id.strip())] += 1

    log.info(
        "Clicks: %d seen, %d kept after train-qid filter (%d unique (query, doc) pairs)",
        n_seen, n_kept, len(click_counts),
    )

    query_to_idx: dict[str, int] = {}
    doc_to_idx:   dict[str, int] = {}
    rows, cols, data = [], [], []

    for (query, doc), count in click_counts.items():
        if query not in query_to_idx:
            query_to_idx[query] = len(query_to_idx)
        if doc not in doc_to_idx:
            doc_to_idx[doc] = len(doc_to_idx)
        rows.append(query_to_idx[query])
        cols.append(doc_to_idx[doc])
        data.append(float(count))

    log.info(
        "Graph: %d queries, %d docs, %d edges",
        len(query_to_idx), len(doc_to_idx), len(data),
    )
    return query_to_idx, doc_to_idx, rows, cols, data


def prune_click_graph(query_to_idx, doc_to_idx, rows, cols, data):
    """
    Apply Craswell et al.'s two-stage pruning iterated to convergence:
      Stage 1: remove docs connected to only 1 query
      Stage 2: remove queries connected to only 1 doc
    Returns pruned graph with contiguous 0-based indices.
    """
    edges: set[tuple[int, int]] = set(zip(rows, cols))

    round_num = 0
    while True:
        round_num += 1
        n_before = len(edges)

        doc_degree: dict[int, int] = defaultdict(int)
        for q, d in edges:
            doc_degree[d] += 1
        singleton_docs = {d for d, deg in doc_degree.items() if deg <= 1}
        edges = {(q, d) for q, d in edges if d not in singleton_docs}

        query_degree: dict[int, int] = defaultdict(int)
        for q, d in edges:
            query_degree[q] += 1
        singleton_queries = {q for q, deg in query_degree.items() if deg <= 1}
        edges = {(q, d) for q, d in edges if q not in singleton_queries}

        n_removed = n_before - len(edges)
        log.info(
            "Pruning round %d: removed %d edges "
            "(%d singleton docs, %d singleton queries)",
            round_num, n_removed, len(singleton_docs), len(singleton_queries),
        )
        if n_removed == 0:
            break

    surviving_queries = sorted({q for q, d in edges})
    surviving_docs    = sorted({d for q, d in edges})

    idx_to_query = {v: k for k, v in query_to_idx.items()}
    idx_to_doc   = {v: k for k, v in doc_to_idx.items()}

    new_query_to_idx = {idx_to_query[q]: i for i, q in enumerate(surviving_queries)}
    new_doc_to_idx   = {idx_to_doc[d]:   i for i, d in enumerate(surviving_docs)}

    old_q_to_new = {q: i for i, q in enumerate(surviving_queries)}
    old_d_to_new = {d: i for i, d in enumerate(surviving_docs)}

    new_rows = [old_q_to_new[q] for q, d in edges]
    new_cols = [old_d_to_new[d] for q, d in edges]
    new_data = [1.0] * len(edges)

    log.info(
        "Pruned graph: %d queries, %d docs, %d edges "
        "(from %d queries, %d docs, %d edges)",
        len(new_query_to_idx), len(new_doc_to_idx), len(edges),
        len(query_to_idx), len(doc_to_idx), len(rows),
    )
    return new_query_to_idx, new_doc_to_idx, new_rows, new_cols, new_data


def build_normalised_adjacency(query_to_idx, doc_to_idx, rows, cols, data):
    """
    Build column-normalised bipartite adjacency matrix in the combined node
    space [queries | docs]. Returns (A_hat, n_q, n_d).
    """
    n_q = len(query_to_idx)
    n_d = len(doc_to_idx)
    N   = n_q + n_d

    r = np.array(rows, dtype=np.int32)
    c = np.array(cols, dtype=np.int32) + n_q  # offset docs into combined space
    w = np.array(data, dtype=np.float32)

    all_rows = np.concatenate([r, c])
    all_cols = np.concatenate([c, r])
    all_data = np.concatenate([w, w])

    A = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(N, N)).tocsr()

    col_sums = np.array(A.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1.0
    A_hat = A @ sp.diags(1.0 / col_sums)

    log.info("Adjacency matrix built: %s, nnz=%d", A_hat.shape, A_hat.nnz)
    return A_hat.tocsr(), n_q, n_d


def iter_corpus():
    """Yield (doc_id, tokens) from CORPUS_PATH (file or directory of .jsonl)."""
    p = Path(CORPUS_PATH)
    files = sorted(p.glob("*.jsonl")) + sorted(p.glob("*.jsonl.gz")) \
        if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in: {CORPUS_PATH}")
    for fpath in files:
        opener = gzip.open if str(fpath).endswith(".gz") else open
        with opener(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj    = json.loads(line)
                doc_id = str(obj.get("id", obj.get("docid", "")))
                text   = obj.get("contents", obj.get("text", ""))
                tokens = tokenize(text)
                if tokens:
                    yield doc_id, tokens

# =============================================================================
# AGGREGATE PPR MODE
# =============================================================================

def ppr_aggregate(A_hat, seed_indices):
    """
    Compute Σ_q π_q exactly by running one power iteration with the
    un-normalized sum of seed vectors as the restart vector.
    By linearity of the PPR recurrence this is mathematically exact.
    Returns pi_total: shape (N,).
    """
    N = A_hat.shape[0]
    e_total          = np.zeros(N, dtype=np.float64)
    e_total[seed_indices] = 1.0
    pi_total = e_total.copy()
    for _ in range(PPR_ITERS):
        pi_total = (1.0 - PPR_ALPHA) * (A_hat @ pi_total) + PPR_ALPHA * e_total
    return pi_total


def generate_weights():
    train_qids = load_train_qids()
    query_to_idx, doc_to_idx, rows, cols, data = build_click_graph(train_qids)
    A_hat, n_q, n_d = build_normalised_adjacency(
        query_to_idx, doc_to_idx, rows, cols, data
    )

    log.info(
        "Running aggregate PPR (alpha=%.2f, iters=%d) over %d train queries",
        PPR_ALPHA, PPR_ITERS, n_q,
    )
    pi_total   = ppr_aggregate(A_hat, np.arange(n_q, dtype=np.int64))
    doc_scores = pi_total[n_q:]

    log.info("PPR complete. Non-zero doc scores: %d / %d",
             (doc_scores > 0).sum(), n_d)

    idx_to_doc       = {v: k for k, v in doc_to_idx.items()}
    doc_id_to_score  = {
        idx_to_doc[j]: float(doc_scores[j])
        for j in range(n_d) if doc_scores[j] > 0
    }

    log.info("Writing term weights to %s", OUTPUT_FILE)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    written = skipped_no_score = skipped_no_tokens = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for doc_id, tokens in tqdm(iter_corpus(), desc="Corpus"):
            score = doc_id_to_score.get(doc_id, 0.0)
            if score <= 0.0:
                skipped_no_score += 1
                continue
            if not tokens:
                skipped_no_tokens += 1
                continue

            if WEIGHT_BY_TF:
                tf: dict[str, int] = defaultdict(int)
                for t in tokens:
                    tf[t] += 1
                total_tf    = float(sum(tf.values()))
                term_weights = {t: c / total_tf for t, c in tf.items()}
            else:
                unique_terms = list(dict.fromkeys(tokens))
                uniform      = 1.0 / len(unique_terms)
                term_weights = {t: uniform for t in unique_terms}

            raw   = {t: w * score for t, w in term_weights.items()}
            total = sum(raw.values())
            for term, prob in raw.items():
                out_f.write(f"-1\t{doc_id}\t{term}\t{prob / total:.8f}\n")
                written += 1

    log.info(
        "Done. Lines written: %d | Docs skipped (no score): %d | (no tokens): %d",
        written, skipped_no_score, skipped_no_tokens,
    )

# =============================================================================
# EXPANSION MODE
# =============================================================================

def generate_expansion_weights():
    """
    Query-pivot co-click expansion + RLM Method 2 marginalization.

    For each pair of documents (d, d′) that share at least one clicking query,
    accumulates sim(d, d′) = Σ_q 1/degree(q). Then for each document takes
    top-k neighbors by sim score and marginalizes their term distributions:

        P(w | d_expanded) = Σ_{d′ ∈ top_k} P(d′|d) · P(w|d′)

    New terms (not in d's original vocabulary) are included in the output.
    """
    train_qids = load_train_qids()
    query_to_idx, doc_to_idx, rows, cols, data = build_click_graph(train_qids)
    query_to_idx, doc_to_idx, rows, cols, data = prune_click_graph(
        query_to_idx, doc_to_idx, rows, cols, data
    )

    log.info("Building query-pivot co-click similarity...")
    query_to_docs: dict[int, list[int]] = defaultdict(list)
    for q_idx, d_idx in zip(rows, cols):
        query_to_docs[q_idx].append(d_idx)

    sim: dict[tuple[int, int], float] = defaultdict(float)
    for q_idx, doc_list in tqdm(query_to_docs.items(), desc="Query pivot"):
        if len(doc_list) < 2:
            continue
        w = 1.0 / len(doc_list)
        for a in range(len(doc_list)):
            for b in range(a + 1, len(doc_list)):
                sim[(doc_list[a], doc_list[b])] += w

    log.info("Co-click pairs accumulated: %d", len(sim))

    doc_neighbors: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for (i, j), score in sim.items():
        doc_neighbors[i].append((score, j))
        doc_neighbors[j].append((score, i))
    del sim

    log.info("Loading corpus into memory for expansion lookup...")
    corpus_map: dict[str, list[str]] = {}
    for doc_id, tokens in iter_corpus():
        corpus_map[doc_id] = tokens
    log.info("Corpus loaded: %d documents", len(corpus_map))

    graph_doc_ids = list(doc_to_idx.keys())
    log.info(
        "Documents in pruned click graph: %d / %d in corpus",
        len(graph_doc_ids), len(corpus_map),
    )
    log.info(
        "Running RLM marginalization (top_k=%d) over %d documents",
        TOP_K, len(graph_doc_ids),
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for j in tqdm(range(len(graph_doc_ids)), desc="RLM expansion"):
            doc_id    = graph_doc_ids[j]
            neighbors = doc_neighbors.get(j)

            if not neighbors:
                skipped += 1
                continue

            top_neighbors = (
                sorted(neighbors, reverse=True)[:TOP_K]
                if len(neighbors) > TOP_K else neighbors
            )
            scores = np.array([s for s, _ in top_neighbors], dtype=np.float64)
            idxs   = [idx for _, idx in top_neighbors]
            scores /= scores.sum()

            expansion_probs: dict[str, float] = defaultdict(float)
            for p_d_prime, neighbor_j in zip(scores, idxs):
                neighbor_tokens = corpus_map.get(graph_doc_ids[neighbor_j])
                if not neighbor_tokens:
                    continue
                tf: dict[str, int] = defaultdict(int)
                for t in neighbor_tokens:
                    tf[t] += 1
                total = float(sum(tf.values()))
                for term, count in tf.items():
                    expansion_probs[term] += p_d_prime * (count / total)

            if not expansion_probs:
                skipped += 1
                continue

            total_prob = sum(expansion_probs.values())
            for term, prob in expansion_probs.items():
                out_f.write(f"-1\t{doc_id}\t{term}\t{prob / total_prob:.8f}\n")
                written += 1

    log.info(
        "Done. Lines written: %d | Docs skipped (no neighbors): %d",
        written, skipped,
    )

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mode_label = "expand" if EXPAND else "aggregate"
    emissions_file = WORK_DIR / f"emissions_clickgraph_{TOKENIZER}_{mode_label}.csv"

    tracker = OfflineEmissionsTracker(
        project_name=f"clickgraph_weights_{TOKENIZER}_{mode_label}",
        country_iso_code=COUNTRY_ISO_CODE,
        output_dir=str(WORK_DIR),
        output_file=emissions_file.name,
        log_level="error",   # suppress codecarbon's own verbose output
        measure_power_secs=30,
    )
    tracker.start()
    log.info("CodeCarbon tracker started (output: %s)", emissions_file)

    try:
        if EXPAND:
            generate_expansion_weights()
        else:
            generate_weights()
    finally:
        # Always stop the tracker even if the script crashes partway through,
        # so the partial energy measurement is written to the CSV.
        emissions = tracker.stop()
        log.info(
            "Energy this run: %.6f kg CO2eq (see %s for full history)",
            emissions, emissions_file,
        )
