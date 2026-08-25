"""
lavrenko_weights.py
==============
Generates per-(doc, term) weights using Relevance-Based Language Models,
Method 2 (Lavrenko & Croft 2001), in the same TSV format expected by
sort_weights.py and ReweightIndex.java:
 
    traversal_idx \t doc_id \t term \t probability
 
All configuration is in the CONFIGURATION section below — edit before running.
 
Output is NOT yet sorted; pipe through sort_weights.py afterwards.
 
Design notes
──────────────────────────────────────────────────────────────────────────────
• Train/test leakage: ORCAS clicks are filtered to query_ids present in
  QRELS_TRAIN_PATH before any weights are computed, mirroring the split used
  for evaluation. Test queries never influence the term weights.
 
• Query-document association: for each document d, ORCAS provides a set of
  train queries Q(d) that users issued when clicking d. These queries drive the
  RLM computation — the method is applied from the queries-that-belong-to-a-
  document direction, not the documents-that-belong-to-a-query direction.
 
• RLM Method 2 (conditional sampling): for each query q in Q(d), the top-N
  BM25 feedback documents are retrieved to form the relevance set Omega. Each
  feedback document is modelled as a Jelinek-Mercer smoothed unigram LM
  (lambda=LAMBDA_SMOOTH). P(w | R, q) is computed in log-space as:
      log P(w|R,q) = log P(w) + Σ_t log P(t|w),  P(t|w) = Σ_d P(Md|w)·P(t|Md)
  The target document d is injected into Omega if it does not appear in the
  top-N hits, ensuring its own term distribution always contributes.
 
• Multi-query aggregation: P(w | R, q) is computed independently for each
  q in Q(d), then aggregated into a single weight per term via AGGREGATION:
    weighted_mean — weights each query's contribution by the BM25 score of d
                    under that query (higher-scoring queries contribute more)
    mean          — unweighted average across queries
    max           — per-term maximum across queries
 
• Tokenisation: all text (corpus, queries, feedback documents from the index)
  has already been through the dataset preprocessor — NFKC-normalised,
  lowercased, regex-tokenised, stopword-removed, and Porter-stemmed — and is
  stored as space-joined tokens. tokenize() therefore splits on whitespace
  rather than re-running the regex, avoiding any risk of non-idempotent
  re-tokenisation.
 
• Parallelisation: the corpus is split into N_WORKERS contiguous chunks.
  Each worker spawns its own JVM and LuceneSearcher (JPype cannot safely
  inherit a JVM across fork/spawn). Contiguous chunks improve LRU cache
  hit rates since topically similar documents tend to share feedback docs.
  Workers write to per-worker partial files; the parent merges them on completion.
 
• Checkpointing: each worker appends completed doc_ids to a checkpoint file
  after every document. On Slurm resume, already-completed docs are skipped
  and the partial output file is appended to. A job hitting the wall-time limit
  can be resubmitted with no repeated work.
 
• Energy tracking: each worker runs its own OfflineEmissionsTracker (codecarbon),
  writing to a per-worker CSV in the checkpoint directory. Each Slurm run
  overwrites the previous CSV. Use the average power draw from the CSV combined
  with known wall-clock times across all Slurm runs to estimate total energy.
 
• traversal_idx is left as -1 in the raw output; sort_weights.py fills it in.
"""

import gc
import json
import logging
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
from codecarbon import OfflineEmissionsTracker
from tqdm import tqdm

# =============================================================================
# CONFIGURATION — edit before running / submitting
# =============================================================================

# --- Input paths ---
TOKENIZER = "anserini"  # "anserini", "symbol" or "alphanum"

WORK_DIR    = Path("..")

MSORCAS_PATH     = WORK_DIR / f"msorcas_preprocessed_{TOKENIZER}.tsv"
ORCAS_PATH       = WORK_DIR / f"orcas_preprocessed_{TOKENIZER}.tsv"
QRELS_TRAIN_PATH = WORK_DIR / f"qrels_{TOKENIZER}_train.tsv"
QRELS_TEST_PATH  = WORK_DIR / f"qrels_{TOKENIZER}_test.tsv"
INDEX_DIR        = WORK_DIR / f"anserini_index_{TOKENIZER}"

# --- DIAGNOSTIC LEAKAGE MODE ---
# WARNING: this trains RLM weights using the TEST queries instead of train.
# This is ONLY valid as a diagnostic upper-bound experiment to check whether
# the reweighting mechanism has any headroom at all. It must NEVER be used
# to produce a result presented as a legitimate evaluation — doing so is
# train/test leakage by construction.
LEAK_TEST_QUERIES = False

# --- Output ---
OUTPUT_FILE = WORK_DIR / f"rlm_weights_{TOKENIZER}_raw.tsv"

# --- Parallelisation ---
N_WORKERS = 8   # set to 1 to disable multiprocessing (e.g. for profiling)

# --- RLM hyperparameters ---
N_FEEDBACK          = 50   # feedback documents per query (paper uses 50)
LAMBDA_SMOOTH       = 0.6  # Jelinek-Mercer lambda (paper uses 0.6)
MAX_QUERIES_PER_DOC = 10   # cap to keep compute tractable in the full run

# Aggregation strategy across queries for a single document.
# "weighted_mean" : average weighted by BM25 score of d under each q
#                   (queries where d ranked higher contribute more)
# "mean"          : unweighted average across queries
# "max"           : per-term maximum across queries
AGGREGATION = "weighted_mean"

# --- BM25 settings (match bm25_baseline.py) ---
BM25_K1 = 0.9
BM25_B  = 0.4

# --- Checkpointing ---
# Partial output and checkpoint files are written to CHECKPOINT_DIR.
# Set to None to use WORK_DIR / "rlm_checkpoints".
CHECKPOINT_DIR: Path | None = None

# --- CodeCarbon ---
# Uses OfflineEmissionsTracker (no internet required on the cluster).
# Each worker writes to its own emissions CSV in the checkpoint directory.
# Each Slurm run overwrites the previous CSV. Use the average power draw
# from the CSV combined with known wall-clock times to estimate total energy.
# NLD = Netherlands (ISO 3166-1 alpha-3 code for your cluster location).
COUNTRY_ISO_CODE = "NLD"

# --- Debug mode ---
# Selects DEBUG_N_DOCS documents that each have >= DEBUG_MIN_QUERIES ORCAS
# queries, sorted by query count descending (richest aggregation signal first).
# Debug runs are always single-process.
DEBUG             = False
DEBUG_N_DOCS      = 20
DEBUG_MIN_QUERIES = 3

# =============================================================================
# LOGGING
# =============================================================================

def make_logger(name: str = __name__) -> logging.Logger:
    """Create a logger that includes the process name in each line."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(processName)s — %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

log = make_logger()

# =============================================================================
# TOKENISATION
# =============================================================================

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours",
    "yourself", "yourselves",
}

def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """
    Split already-tokenized text on whitespace.
 
    All text this function ever sees (corpus documents, ORCAS queries,
    feedback document text fetched from the index) has already been through
    the dataset preprocessor — NFKC normalised, lowercased, regex-tokenized,
    stopword-removed, and Porter-stemmed — and stored as space-joined tokens.
    Re-running the original tokenizing regex over that text would be a
    redundant, no-op-in-practice re-parse (and a latent risk if the regex
    is ever non-idempotent on some token), so this just splits on whitespace.
 
    The remove_stopwords flag is kept for interface compatibility with the
    rest of the pipeline, but is normally a no-op since stopwords were
    already stripped during preprocessing.
    """
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
 
 
# =============================================================================
# DATA LOADING
# =============================================================================
 
def load_corpus(path: Path) -> dict[str, str]:
    log.info(f"Loading corpus from {path}")
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["docid", "title", "body"],
        dtype=str, quoting=3, on_bad_lines="skip",
    )
    df["title"] = df["title"].fillna("")
    df["body"]  = df["body"].fillna("")
    corpus = dict(zip(df["docid"], df["title"] + " " + df["body"]))
    log.info(f"Loaded {len(corpus):,} documents")
    return corpus
 
 
def load_train_qids(path: Path) -> set[str]:
    log.info(f"Loading train qids from {path}")
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["qid", "query", "docid"],
        dtype=str, quoting=3, on_bad_lines="skip",
    )
    qids = set(df["qid"].dropna().unique())
    log.info(f"Found {len(qids):,} unique train qids")
    return qids
 
 
def load_orcas(path: Path, train_qids: set[str]) -> dict[str, list[str]]:
    log.info(f"Loading ORCAS from {path}")
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["qid", "query", "docid"],
        dtype=str, quoting=3, on_bad_lines="skip",
    )
    df["query"] = df["query"].fillna("").str.strip()
    df = df[df["query"] != ""].reset_index(drop=True)
 
    n_before = len(df)
    df = df[df["qid"].isin(train_qids)].reset_index(drop=True)
    log.info(f"Retained {len(df):,} / {n_before:,} rows after filtering to train qids")
 
    doc_to_queries: dict[str, list[str]] = (
        df.groupby("docid")["query"].apply(list).to_dict()
    )
    total_pairs = sum(len(v) for v in doc_to_queries.values())
    log.info(
        f"Loaded {total_pairs:,} train query-doc pairs "
        f"across {len(doc_to_queries):,} documents"
    )
    return doc_to_queries
 
 
# =============================================================================
# DEBUG SUBSET SELECTION
# =============================================================================
 
def select_debug_subset(
    doc_to_queries: dict[str, list[str]],
    corpus: dict[str, str],
    n_docs: int,
    min_queries: int,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    candidates = [
        (doc_id, queries)
        for doc_id, queries in doc_to_queries.items()
        if doc_id in corpus and len(queries) >= min_queries
    ]
    if not candidates:
        raise ValueError(
            f"No documents with >= {min_queries} queries. "
            f"Lower DEBUG_MIN_QUERIES or check data files."
        )
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    selected = candidates[:n_docs]
 
    counts = [len(q) for _, q in selected]
    log.info(
        f"Debug subset: {len(selected)} docs | "
        f"queries/doc — min {min(counts)}, max {max(counts)}, "
        f"mean {sum(counts)/len(counts):.1f}"
    )
    for doc_id, queries in selected[:5]:
        log.info(f"  {doc_id} ({len(queries)} queries) e.g. '{queries[0][:70]}'")
    if len(selected) > 5:
        log.info(f"  ... and {len(selected) - 5} more")
 
    return (
        {doc_id: corpus[doc_id] for doc_id, _ in selected},
        {doc_id: queries for doc_id, queries in selected},
    )
 
 
# =============================================================================
# COLLECTION STATISTICS
# =============================================================================
 
def build_collection_stats(corpus: dict[str, str]) -> tuple[dict[str, int], int]:
    log.info("Building collection statistics …")
    cf: dict[str, int] = defaultdict(int)
    total = 0
    for text in tqdm(corpus.values(), desc="collection stats", unit="doc"):
        for tok in tokenize(text, remove_stopwords=True):
            cf[tok] += 1
            total += 1
    log.info(f"Vocabulary: {len(cf):,} terms | Total tokens: {total:,}")
    return dict(cf), total
 
 
# =============================================================================
# SMOOTHED UNIGRAM MODEL
# =============================================================================
 
def build_unigram_model(
    tokens: list[str],
    collection_freq: dict[str, int],
    collection_size: int,
    lam: float,
) -> dict[str, float]:
    if not tokens:
        return {}
    doc_len = len(tokens)
    tf: dict[str, int] = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    model: dict[str, float] = {}
    for term, count in tf.items():
        bg = collection_freq.get(term, 0) / collection_size if collection_size > 0 else 0.0
        model[term] = lam * (count / doc_len) + (1 - lam) * bg
    return model
 
 
def bg_prob(term: str, collection_freq: dict[str, int], collection_size: int) -> float:
    if collection_size <= 0:
        return 1e-10
    return collection_freq.get(term, 0.5) / collection_size
 
 
# =============================================================================
# RLM METHOD 2
# =============================================================================
 
def make_doc_text_fetcher(searcher):
    """
    Return an LRU-cached function that fetches stored document text via
    searcher.doc().  Each unique docid triggers exactly one JNI call;
    subsequent lookups are served from the in-process cache.
    """
    @lru_cache(maxsize=8192)
    def fetch(docid: str) -> str:
        try:
            stored = searcher.doc(docid)
            if stored is None:
                return ""
            raw_json = stored.raw()
            if not raw_json:
                return ""
            try:
                return json.loads(raw_json).get("contents", "")
            except (json.JSONDecodeError, ValueError):
                return raw_json
        except Exception as exc:
            log.warning(f"Could not fetch doc {docid}: {exc}")
            return ""
    return fetch
 
 
def rlm_method2(
    query_text: str,
    doc_id: str,
    doc_term_set: set[str],
    hits: list,
    fetch_doc_text,
    collection_freq: dict[str, int],
    collection_size: int,
    lam: float,
) -> tuple[dict[str, float], float]:
    """
    Compute P(w | R, q) for each term w in doc_term_set using Method 2.
 
    Method 2 (conditional sampling):
        log P(w|R,q) = log P(w) + Σ_t log P(t|w)
        P(t|w) = Σ_d [P(w|Md)/Z_w] · P(t|Md)
 
    hits are pre-fetched by the caller — no redundant searcher.search() here.
    fetch_doc_text is LRU-cached — each docid costs one JNI call total.
    """
    if not hits:
        return {}, 0.0
 
    feedback_models: list[dict[str, float]] = []
    doc_bm25_score = 0.0
 
    for hit in hits:
        raw_text = fetch_doc_text(hit.docid)
        fb_tokens = tokenize(raw_text, remove_stopwords=True)
        model = build_unigram_model(fb_tokens, collection_freq, collection_size, lam)
        if model:
            feedback_models.append(model)
        if hit.docid == doc_id and doc_bm25_score == 0.0:
            doc_bm25_score = float(hit.score)
 
    if not feedback_models:
        return {}, doc_bm25_score
 
    query_tokens = tokenize(query_text, remove_stopwords=True)
    if not query_tokens:
        return {}, doc_bm25_score
 
    n_models = len(feedback_models)
    total_cf = sum(collection_freq.get(w, 0.5) for w in doc_term_set)
    if total_cf <= 0:
        total_cf = 1.0
 
    weights: dict[str, float] = {}
 
    for w in doc_term_set:
        p_w = collection_freq.get(w, 0.5) / total_cf
        p_w_per_model = [
            m.get(w, bg_prob(w, collection_freq, collection_size))
            for m in feedback_models
        ]
        z_w = sum(p_w_per_model)
        if z_w <= 0:
            weights[w] = 0.0
            continue
 
        log_score = math.log(p_w)
        for t in query_tokens:
            p_t_given_w = sum(
                (p_w_per_model[i] / z_w)
                * feedback_models[i].get(t, bg_prob(t, collection_freq, collection_size))
                for i in range(n_models)
            )
            log_score += math.log(max(p_t_given_w, 1e-10))
 
        weights[w] = math.exp(log_score)
 
    return weights, doc_bm25_score
 
 
# =============================================================================
# AGGREGATION
# =============================================================================
 
def aggregate(
    per_query: list[tuple[float, dict[str, float]]],
    doc_term_set: set[str],
    strategy: str,
) -> dict[str, float]:
    if not per_query:
        return {}
 
    result: dict[str, float] = {t: 0.0 for t in doc_term_set}
 
    if strategy == "max":
        for _, weights in per_query:
            for term in doc_term_set:
                result[term] = max(result[term], weights.get(term, 0.0))
 
    elif strategy == "mean":
        n = len(per_query)
        for _, weights in per_query:
            for term in doc_term_set:
                result[term] += weights.get(term, 0.0) / n
 
    elif strategy == "weighted_mean":
        total_score = sum(s for s, _ in per_query)
        if total_score <= 0:
            return aggregate(per_query, doc_term_set, "mean")
        for score, weights in per_query:
            frac = score / total_score
            for term in doc_term_set:
                result[term] += frac * weights.get(term, 0.0)
 
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")
 
    return result
 
 
# =============================================================================
# CHECKPOINTING
# =============================================================================
 
def load_checkpoint(checkpoint_file: Path) -> set[str]:
    """
    Read the set of already-completed doc_ids from a checkpoint file.
    Returns an empty set if the file does not exist.
    """
    if not checkpoint_file.exists():
        return set()
    with open(checkpoint_file, encoding="utf-8") as f:
        completed = {line.strip() for line in f if line.strip()}
    return completed
 
 
def append_checkpoint(checkpoint_file: Path, doc_id: str) -> None:
    """Append a single doc_id to the checkpoint file (one per line)."""
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(doc_id + "\n")
 
 
# =============================================================================
# WORKER FUNCTION  (runs in a separate process)
# =============================================================================
 
def worker(
    worker_id: int,
    doc_ids_chunk: list[str],
    corpus: dict[str, str],
    doc_to_queries: dict[str, list[str]],
    index_dir: Path,
    partial_file: Path,
    checkpoint_file: Path,
    collection_freq: dict[str, int],
    collection_size: int,
    n_feedback: int,
    lam: float,
    max_queries_per_doc: int,
    aggregation: str,
) -> int:
    """
    Process one chunk of documents.  Returns the number of (doc, term) pairs
    written.  Designed to be called via multiprocessing.Process.
 
    Key design decisions:
      - LuceneSearcher is initialised here, inside the worker process.
        JPype cannot safely fork a JVM, so the searcher must never be created
        in the parent process when using multiprocessing.
      - Checkpoint is loaded at startup; completed docs are skipped immediately.
      - Output file is opened in append mode so resumed jobs don't overwrite
        prior work.  The partial file for a fresh worker is created empty by
        the parent before launching workers (so the header is written once).
    """
    # Each worker needs its own logger (logging is not fork-safe on all platforms)
    wlog = make_logger(f"worker-{worker_id}")
    wlog.info(f"Worker {worker_id} starting — {len(doc_ids_chunk):,} docs in chunk")
 
    # Load checkpoint: skip docs already done in a previous run
    completed = load_checkpoint(checkpoint_file)
    remaining = [d for d in doc_ids_chunk if d not in completed]
    if completed:
        wlog.info(
            f"Worker {worker_id} resuming: {len(completed):,} already done, "
            f"{len(remaining):,} remaining"
        )
 
    if not remaining:
        wlog.info(f"Worker {worker_id} — nothing to do, exiting")
        return 0
 
    # Initialise searcher inside the worker (never in the parent process)
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(BM25_K1, BM25_B)
    fetch_doc_text = make_doc_text_fetcher(searcher)
 
    # CodeCarbon tracker — one per worker, one CSV file per worker.
    # Each Slurm run overwrites the previous CSV. Use the per-worker CSVs
    # to read power draw and combine with known wall-clock times to estimate
    # total energy across all runs.
    emissions_file = checkpoint_file.parent / f"emissions_worker_{worker_id}.csv"
    tracker = OfflineEmissionsTracker(
        project_name=f"rlm_weights_worker_{worker_id}",
        country_iso_code=COUNTRY_ISO_CODE,
        output_dir=str(checkpoint_file.parent),
        output_file=f"emissions_worker_{worker_id}.csv",
        log_level="error",
        measure_power_secs=30,
    )
    tracker.start()
    wlog.info(f"Worker {worker_id} — CodeCarbon tracker started (output: {emissions_file})")

    n_written = 0
    n_skipped = 0
 
    try:
        # tqdm position=worker_id gives each worker its own line in the terminal
        with open(partial_file, "a", encoding="utf-8") as out:
            for doc_id in tqdm(
                remaining,
                desc=f"worker-{worker_id}",
                position=worker_id,
                leave=True,
            ):
                doc_text = corpus.get(doc_id, "")
                doc_term_set = set(tokenize(doc_text, remove_stopwords=True))
 
                if not doc_term_set:
                    wlog.warning(f"Empty term set: {doc_id}")
                    append_checkpoint(checkpoint_file, doc_id)
                    n_skipped += 1
                    continue
 
                queries = doc_to_queries.get(doc_id, [])
                if not queries:
                    wlog.warning(f"No queries: {doc_id}")
                    append_checkpoint(checkpoint_file, doc_id)
                    n_skipped += 1
                    continue
 
                if len(queries) > max_queries_per_doc:
                    queries = queries[:max_queries_per_doc]
 
                per_query_weights: list[tuple[float, dict[str, float]]] = []
 
                for query_text in queries:
                    try:
                        hits = searcher.search(query_text, k=n_feedback)
                    except Exception as exc:
                        wlog.warning(f"Search failed '{query_text[:50]}': {exc}")
                        continue
 
                    weights, doc_score = rlm_method2(
                        query_text=query_text,
                        doc_id=doc_id,
                        doc_term_set=doc_term_set,
                        hits=hits,
                        fetch_doc_text=fetch_doc_text,
                        collection_freq=collection_freq,
                        collection_size=collection_size,
                        lam=lam,
                    )
                    if weights:
                        per_query_weights.append((doc_score, weights))
 
                if not per_query_weights:
                    wlog.warning(f"No valid weights: {doc_id}")
                    append_checkpoint(checkpoint_file, doc_id)
                    n_skipped += 1
                    continue
 
                final = aggregate(per_query_weights, doc_term_set, aggregation)
 
                for term, weight in final.items():
                    if weight > 0.0:
                        out.write(f"{doc_id}\t{term}\t{weight:.10f}\n")
                        n_written += 1
 
                # Checkpoint after each document so any crash loses at most one doc
                append_checkpoint(checkpoint_file, doc_id)
                gc.collect()


 
    finally:
        # Always stop the tracker so the CSV is written even if the worker
        # exits early. On SIGKILL the CSV may be incomplete — use wall-clock
        # time and average power from the CSV to estimate total energy.
        try:
            emissions = tracker.stop()
        except Exception:
            emissions = 0.0
        wlog.info(f"Worker {worker_id} — energy this run: {emissions:.6f} kg CO2eq")
 
    wlog.info(
        f"Worker {worker_id} done — "
        f"{len(remaining) - n_skipped:,} processed, "
        f"{n_skipped:,} skipped, "
        f"{n_written:,} (doc, term) pairs written"
    )
    return n_written
 
 
# =============================================================================
# MERGE PARTIAL FILES
# =============================================================================
 
def merge_partial_files(partial_files: list[Path], output_file: Path) -> None:
    """
    Concatenate all worker partial files into the final output file.
    Writes the header once, then streams each partial file line by line
    to avoid loading everything into memory.
    """
    log.info(f"Merging {len(partial_files)} partial files into {output_file}")
    total_rows = 0
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("doc_id\tterm\tprobability\n")
        for pf in partial_files:
            if not pf.exists():
                log.warning(f"Partial file missing: {pf}")
                continue
            with open(pf, encoding="utf-8") as src:
                for line in src:
                    out.write(line)
                    total_rows += 1
    log.info(f"Merge complete — {total_rows:,} data rows written to {output_file}")
 
 
# =============================================================================
# EMISSIONS SUMMARY
# =============================================================================
 
def summarise_emissions(checkpoint_dir: Path, n_workers: int) -> None:
    """
    After all workers finish, read the last row of each worker's emissions CSV
    and log the energy consumed in this Slurm run. Each CSV is overwritten on
    each run — use wall-clock time and average power to estimate total energy
    across multiple runs.
    """
    total_energy_kwh = 0.0
    total_co2_kg     = 0.0
    n_files_found    = 0

    for i in range(n_workers):
        emissions_file = checkpoint_dir / f"emissions_worker_{i}.csv"
        if not emissions_file.exists():
            log.warning(f"Emissions file not found for worker {i}: {emissions_file}")
            continue
        try:
            df = pd.read_csv(emissions_file)
            worker_kwh = float(df["energy_consumed"].iloc[-1])
            worker_co2 = float(df["emissions"].iloc[-1])
            log.info(
                f"  Worker {i}: {worker_kwh:.6f} kWh, {worker_co2:.6f} kg CO2eq "
                f"(this run only)"
            )
            total_energy_kwh += worker_kwh
            total_co2_kg     += worker_co2
            n_files_found    += 1
        except Exception as exc:
            log.warning(f"Could not read emissions file for worker {i}: {exc}")

    log.info(
        f"=== TOTAL ENERGY THIS RUN ({n_files_found}/{n_workers} workers) === "
        f"{total_energy_kwh:.6f} kWh | {total_co2_kg:.6f} kg CO2eq"
    )
 
 
# =============================================================================
# PARALLEL ORCHESTRATOR
# =============================================================================
 
def run_parallel(
    corpus: dict[str, str],
    doc_to_queries: dict[str, list[str]],
    index_dir: Path,
    output_file: Path,
    checkpoint_dir: Path,
    n_workers: int,
    collection_freq: dict[str, int],
    collection_size: int,
    n_feedback: int,
    lam: float,
    max_queries_per_doc: int,
    aggregation: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoint directory: {checkpoint_dir}")
 
    # Split doc_ids into contiguous chunks — one per worker.
    # Contiguous (not interleaved) so that each worker's LRU cache is warmer:
    # documents that are topically similar tend to share feedback documents.
    all_doc_ids = list(corpus.keys())
    chunk_size = math.ceil(len(all_doc_ids) / n_workers)
    chunks = [
        all_doc_ids[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n_workers)
        if i * chunk_size < len(all_doc_ids)
    ]
    actual_workers = len(chunks)
    log.info(
        f"Splitting {len(all_doc_ids):,} docs across {actual_workers} workers "
        f"(chunk size ~{chunk_size:,})"
    )
 
    partial_files = [checkpoint_dir / f"partial_{i}.tsv" for i in range(actual_workers)]
    checkpoint_files = [checkpoint_dir / f"checkpoint_{i}.txt" for i in range(actual_workers)]
 
    # Create partial files that don't exist yet (append mode in workers
    # requires the file to exist; we create them empty here)
    for pf in partial_files:
        if not pf.exists():
            pf.touch()
 
    # Launch workers
    processes = []
    for i in range(actual_workers):
        p = mp.Process(
            target=worker,
            name=f"worker-{i}",
            kwargs=dict(
                worker_id=i,
                doc_ids_chunk=chunks[i],
                corpus=corpus,
                doc_to_queries=doc_to_queries,
                index_dir=index_dir,
                partial_file=partial_files[i],
                checkpoint_file=checkpoint_files[i],
                collection_freq=collection_freq,
                collection_size=collection_size,
                n_feedback=n_feedback,
                lam=lam,
                max_queries_per_doc=max_queries_per_doc,
                aggregation=aggregation,
            ),
        )
        p.start()
        processes.append(p)
        log.info(f"Launched worker-{i} (pid {p.pid}) — {len(chunks[i]):,} docs")
 
    # Wait for all workers to finish
    failed = []
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            log.error(f"Worker-{i} exited with code {p.exitcode}")
            failed.append(i)
        else:
            log.info(f"Worker-{i} finished successfully")
 
    if failed:
        log.error(
            f"{len(failed)} worker(s) failed: {failed}. "
            f"Partial files are preserved in {checkpoint_dir} for inspection. "
            f"Fix the issue and rerun — checkpoints will skip completed docs."
        )
        sys.exit(1)
 
    # Merge and clean up
    merge_partial_files(partial_files, output_file)
 
    # Summarise energy across all workers before cleaning up their files
    log.info("=== Energy consumption summary ===")
    summarise_emissions(checkpoint_dir, actual_workers)
 
    log.info(f"All done. Final output: {output_file}")
 
 
# =============================================================================
# SINGLE-PROCESS PATH  (debug mode or N_WORKERS=1)
# =============================================================================
 
def run_single(
    corpus: dict[str, str],
    doc_to_queries: dict[str, list[str]],
    index_dir: Path,
    output_file: Path,
    checkpoint_dir: Path,
    collection_freq: dict[str, int],
    collection_size: int,
    n_feedback: int,
    lam: float,
    max_queries_per_doc: int,
    aggregation: str,
) -> None:
    """
    Single-process version used for debug runs and N_WORKERS=1.
    Also supports checkpointing so debug runs can be resumed.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    partial_file    = checkpoint_dir / "partial_0.tsv"
    checkpoint_file = checkpoint_dir / "checkpoint_0.txt"
 
    if not partial_file.exists():
        partial_file.touch()
 
    worker(
        worker_id=0,
        doc_ids_chunk=list(corpus.keys()),
        corpus=corpus,
        doc_to_queries=doc_to_queries,
        index_dir=index_dir,
        partial_file=partial_file,
        checkpoint_file=checkpoint_file,
        collection_freq=collection_freq,
        collection_size=collection_size,
        n_feedback=n_feedback,
        lam=lam,
        max_queries_per_doc=max_queries_per_doc,
        aggregation=aggregation,
    )
 
    merge_partial_files([partial_file], output_file)
    log.info(f"Done. Output: {output_file}")
 
 
# =============================================================================
# ENTRY POINT
# =============================================================================
 
def main() -> None:
    # Resolve checkpoint directory
    ckpt_dir = CHECKPOINT_DIR if CHECKPOINT_DIR is not None else WORK_DIR / "rlm_checkpoints"
 
    output_file = OUTPUT_FILE
    if DEBUG:
        log.info("=== DEBUG MODE (single-process) ===")
        output_file = WORK_DIR / "rlm_weights.debug.tsv"
        ckpt_dir    = WORK_DIR / "rlm_checkpoints_debug"
 
    # Load shared data in the main process
    corpus         = load_corpus(MSORCAS_PATH)
    train_qids     = load_train_qids(QRELS_TRAIN_PATH)
    doc_to_queries = load_orcas(ORCAS_PATH, train_qids)
 
    if DEBUG:
        corpus, doc_to_queries = select_debug_subset(
            doc_to_queries=doc_to_queries,
            corpus=corpus,
            n_docs=DEBUG_N_DOCS,
            min_queries=DEBUG_MIN_QUERIES,
        )
 
    collection_freq, collection_size = build_collection_stats(corpus)
 
    shared_kwargs = dict(
        corpus=corpus,
        doc_to_queries=doc_to_queries,
        index_dir=INDEX_DIR,
        output_file=output_file,
        checkpoint_dir=ckpt_dir,
        collection_freq=collection_freq,
        collection_size=collection_size,
        n_feedback=N_FEEDBACK,
        lam=LAMBDA_SMOOTH,
        max_queries_per_doc=MAX_QUERIES_PER_DOC,
        aggregation=AGGREGATION,
    )
 
    if DEBUG or N_WORKERS <= 1:
        run_single(**shared_kwargs)
    else:
        run_parallel(n_workers=N_WORKERS, **shared_kwargs)
 
 
if __name__ == "__main__":
    # Required on some platforms (e.g. macOS) to avoid fork-related crashes.
    # On Linux (your cluster) this is a no-op but it's good practice.
    mp.set_start_method("spawn", force=True)
    main()
