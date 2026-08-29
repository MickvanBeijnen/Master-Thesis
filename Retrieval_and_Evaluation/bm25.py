"""
BM25 — Anserini / Pyserini
=====================================
Converted from master-thesis-anserini.ipynb for use on a Slurm cluster.

Edit the CONFIGURATION section below before submitting, then run via:
    sbatch submit.sh
or directly:
    python bm25.py

Train/test split strategy
--------------------------
The split is performed at the *query* level (unique qids), not the document
level.  The index is built on all documents regardless — what we are
evaluating is whether the model generalises to queries it has not seen during
any training/tuning phase.  Retrieval is run separately for both splits so
the result files can be used independently downstream.
"""

import gc
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

# Input data, select which tokenizer to use
TOKENIZER = "anserini"      # "anserini", "symbol" or "alphanum"

# Working directories (created automatically if they don't exist)
WORK_DIR          = Path("..")
CORPUS_DIR        = WORK_DIR / f"corpus_{TOKENIZER}"
INDEX_DIR         = WORK_DIR / f"anserini_index_{TOKENIZER}"
CORPUS_FILE       = CORPUS_DIR / "msorcas.jsonl"

MSORCAS_PATH     = WORK_DIR / f"msorcas_preprocessed_{TOKENIZER}.tsv"
ORCAS_PATH       = WORK_DIR / f"orcas_preprocessed_{TOKENIZER}.tsv"
QRELS_TRAIN_PATH = WORK_DIR / f"qrels_{TOKENIZER}_train.tsv"

RESULTS_DIR         = WORK_DIR / f"results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reweighted index directories — produced by ReweightIndex.java
# Set to None to skip retrieval over that variant
INDEX_DIR_CONSTANT        = WORK_DIR / f"index_constant_{TOKENIZER}"
INDEX_DIR_RANDOM          = WORK_DIR / f"index_random_{TOKENIZER}"
INDEX_DIR_WEIGHTED_RANDOM = WORK_DIR / f"index_weighted_random_{TOKENIZER}"

# RLM alpha variants — any index matching index_rlm_{TOKENIZER}_alpha_* is
# auto-discovered. Built by ReweightIndex.java in rlm_multi mode, e.g.:
#   java ... ReweightIndex <input> <workdir> rlm_multi <weights> 1.0,0.75,0.5,0.25,0.0
# No changes needed here when you add or remove alpha values.

# Number of retrieval workers to run in parallel.
# Each worker loads one index and searches all test queries independently.
# Set to 1 to disable parallelism (useful for debugging).
# Do not exceed the number of CPU cores available on your node.
RETRIEVAL_WORKERS = 1

# Split settings
TEST_SIZE    = 10_000
RANDOM_STATE = 42

# Retrieval settings
BM25_K1       = 0.9
BM25_B        = 0.4
NUM_RESULTS   = 10
BATCH_SIZE    = 5_000
INDEX_THREADS = 4

# Set to True to cap the test run at 1k queries (handy for quick testing)
DEBUG_SAMPLE = False

# =============================================================================
# SETUP
# =============================================================================
 
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
 
print("Config OK")
print(f"  MSORCAS    : {MSORCAS_PATH}")
print(f"  ORCAS      : {ORCAS_PATH}")
print(f"  Work dir   : {WORK_DIR}")
print(f"  Test size  : {TEST_SIZE:,} queries")
print(f"  Random seed: {RANDOM_STATE}")
 
# =============================================================================
# IMPORTS
# =============================================================================
 
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
 
STOPWORDS = set(stopwords.words("english"))
STEMMER   = PorterStemmer()
 
print("Imports OK")
 
# =============================================================================
# DATA LOADING
# =============================================================================
 
print("Loading msorcas...")
msorcas = pd.read_csv(
    MSORCAS_PATH,
    sep="\t",
    header=None,
    names=["docid", "title", "body"],
    dtype=str,
    quoting=3,
    on_bad_lines="skip",
)
print(f"Loaded msorcas: {len(msorcas):,} documents")
print(msorcas.head())
 
print("\nLoading ORCAS...")
orcas = pd.read_csv(
    ORCAS_PATH,
    sep="\t",
    header=None,
    names=["qid", "query", "docid"],
    dtype=str,
    quoting=3,
    on_bad_lines="skip",
)
print(f"Loaded ORCAS: {len(orcas):,} entries")
print(orcas.head())
 
# =============================================================================
# DATA PREPROCESSING
# =============================================================================
 
tqdm.pandas()
 
msorcas["title"] = msorcas["title"].fillna("")
msorcas["body"]  = msorcas["body"].fillna("")
msorcas["all"]   = msorcas["title"] + " " + msorcas["body"]
 
orcas["query"] = orcas["query"].fillna("").astype(str).str.strip()
orcas = orcas[orcas["query"] != ""].reset_index(drop=True)
 
print(msorcas[["docid", "title", "body", "all"]].head())
print("Unique docs in msorcas:", msorcas["docid"].nunique())
print("Unique docs in orcas  :", orcas["docid"].nunique())
print("Unique queries        :", orcas["qid"].nunique())
 
missing_docs = set(orcas["docid"]) - set(msorcas["docid"])
print("Missing docs referenced by ORCAS:", len(missing_docs))
 
# =============================================================================
# CREATE MAPPINGS
# =============================================================================
 
doc_to_queries = orcas.groupby("docid")["query"].apply(list).to_dict()
query_to_docs  = orcas.groupby("query")["docid"].apply(list).to_dict()
doc_to_text    = dict(zip(msorcas["docid"], msorcas["all"]))
 
# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================
 
all_qids = orcas["qid"].unique()
print(f"\nTotal unique qids: {len(all_qids):,}")
 
SPLIT_FILE = WORK_DIR / "split_test_qids.txt"
 
if SPLIT_FILE.exists():
    print(f"Loading existing split from {SPLIT_FILE} ...")
    test_qids  = set(pd.read_csv(SPLIT_FILE, dtype=str, header=None)[0].tolist())
    train_qids = set(orcas["qid"].astype(str).unique()) - test_qids
    print(f"Loaded split — train: {len(train_qids):,}, test: {len(test_qids):,}")
else:
    print(f"Generating new split (seed={RANDOM_STATE}) and saving to {SPLIT_FILE} ...")
    test_qids  = set(pd.Series(all_qids).sample(n=TEST_SIZE, random_state=RANDOM_STATE).astype(str))
    train_qids = set(orcas["qid"].astype(str).unique()) - test_qids
    pd.Series(sorted(test_qids)).to_csv(SPLIT_FILE, index=False, header=False)
    print(f"Split saved — train: {len(train_qids):,}, test: {len(test_qids):,}")
 
orcas_train = orcas[orcas["qid"].astype(str).isin(train_qids)].reset_index(drop=True)
orcas_test  = orcas[orcas["qid"].astype(str).isin(test_qids)].reset_index(drop=True)
 
print(f"Train qrels rows: {len(orcas_train):,}")
print(f"Test  qrels rows: {len(orcas_test):,}")
 
orcas_train.to_csv(WORK_DIR / f"qrels_{TOKENIZER}_train.tsv", sep="\t", index=False, header=False)
orcas_test.to_csv( WORK_DIR / f"qrels_{TOKENIZER}_test.tsv",  sep="\t", index=False, header=False)
print(f"Qrels saved to {WORK_DIR}/qrels_{TOKENIZER}_train.tsv and qrels_{TOKENIZER}_test.tsv")
 
# =============================================================================
# BUILD QUERY TOPIC TABLES
# =============================================================================
 
def build_topics(orcas_split: pd.DataFrame) -> pd.DataFrame:
    """Return a deduplicated topic table with a tokenised query column."""
    topics = orcas_split[["qid", "query"]].drop_duplicates().copy()
    topics["qid"]        = topics["qid"].astype(str)
    topics["query"]      = topics["query"].fillna("").astype(str).str.strip()
    topics["query_toks"] = topics["query"].apply(
        lambda q: {tok: float(cnt) for tok, cnt in Counter(q.split()).items()}
    )
    return topics.reset_index(drop=True)
 
topics_train = build_topics(orcas_train)
topics_test  = build_topics(orcas_test)
 
print(f"\nTopic table sizes — train: {len(topics_train):,}, test: {len(topics_test):,}")
 
if DEBUG_SAMPLE:
    topics_test = topics_test.sample(1_000, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"DEBUG_SAMPLE=True: capping test run at {len(topics_test):,} queries.")
 
# =============================================================================
# BM25 BASELINE — ANSERINI
# Step 1 — Write corpus to JSONL
# =============================================================================
 
if CORPUS_FILE.exists():
    print(f"\nCorpus file already exists at {CORPUS_FILE}, skipping write.")
else:
    print(f"\nWriting {len(msorcas):,} documents to {CORPUS_FILE} ...")
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for row in tqdm(msorcas.itertuples(index=False), total=len(msorcas)):
            doc = {"id": row.docid, "contents": row.all}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print("Done writing corpus.")
 
# =============================================================================
# Step 2 — Build the Anserini index
# =============================================================================
 
index_complete = any(INDEX_DIR.glob("segments*")) if INDEX_DIR.exists() else False
if index_complete:
    print(f"\nComplete index found at {INDEX_DIR}, skipping indexing.")
else:
    if INDEX_DIR.exists():
        print(f"Removing incomplete index at {INDEX_DIR} ...")
        shutil.rmtree(INDEX_DIR)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection",  "JsonCollection",
        "--input",       str(CORPUS_DIR),
        "--index",       str(INDEX_DIR),
        "--generator",   "DefaultLuceneDocumentGenerator",
        "--threads",     str(INDEX_THREADS),
        "--pretokenized",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Indexing complete.")
 
# =============================================================================
# Step 3 — Create the searcher (used only to verify the index is readable)
# =============================================================================
 
from pyserini.search.lucene import LuceneSearcher
 
searcher = LuceneSearcher(str(INDEX_DIR))
searcher.set_bm25(BM25_K1, BM25_B)
 
print(f"\nSearcher ready. BM25 k1={BM25_K1}, b={BM25_B}")
print(f"Index contains {searcher.num_docs:,} documents.")
 
# =============================================================================
# Step 4 — Retrieval worker
#
# Runs in a separate spawned process with its own clean JVM.
# All config values are passed explicitly — spawned processes do not inherit
# the parent's module-level globals.
# Output schema: qid | query | docno | rank | score | query_toks
# =============================================================================
 
def _retrieval_worker(args: tuple) -> str:
    """
    Worker function executed in a separate spawned process.
    Opens a LuceneSearcher for the given index and runs all test queries.
    Returns a status message when done.
    """
    variant_name, index_dir_str, output_path_str, topics_dict, work_dir_str, \
        bm25_k1, bm25_b, batch_size, num_results = args
 
    from pyserini.search.lucene import LuceneSearcher
    import pandas as pd
    import gc
    from pathlib import Path
 
    topics      = pd.DataFrame(topics_dict)
    output_path = Path(output_path_str)
 
    searcher = LuceneSearcher(index_dir_str)
    searcher.set_bm25(bm25_k1, bm25_b)
 
    total       = len(topics)
    first_batch = True
 
    if output_path.exists():
        output_path.unlink()
 
    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        batch = topics.iloc[start:end]
 
        rows = []
        for _, row in batch.iterrows():
            hits = searcher.search(row["query"], k=num_results)
            for rank, hit in enumerate(hits, start=1):
                rows.append({
                    "qid":        row["qid"],
                    "query":      row["query"],
                    "docno":      hit.docid,
                    "rank":       rank,
                    "score":      hit.score,
                    "query_toks": str(row["query_toks"]),
                })
 
        batch_df = pd.DataFrame(rows)
        batch_df.to_csv(
            output_path,
            sep="\t",
            index=False,
            header=first_batch,
            mode="w" if first_batch else "a",
        )
 
        first_batch = False
        del batch, batch_df, rows
        gc.collect()
 
    return f"[{variant_name}] Done. Saved to {output_path}"

 
# =============================================================================
# Step 5 — Build variant list and run retrieval in parallel
#
# Fixed variants are listed explicitly. RLM and click-graph (CG) alpha
# variants are auto-discovered by globbing for index_rlm_{TOKENIZER}_*alpha_*
# and index_cg_{TOKENIZER}_*alpha_* respectively, so no code changes are
# needed when alpha values or MAX_VAL change.
# =============================================================================

fixed_variants = [
    (f"baseline_{TOKENIZER}", INDEX_DIR),
]

# Debug variants — produced by constant/random/weighted_random modes in
# ReweightIndex.java. These are for pipeline verification only and are not
# part of actual experiments. Set any to None to skip retrieval for that variant.
debug_variants = [
    (f"constant_{TOKENIZER}",        INDEX_DIR_CONSTANT),
    (f"random_{TOKENIZER}",          INDEX_DIR_RANDOM),
    (f"weighted_random_{TOKENIZER}", INDEX_DIR_WEIGHTED_RANDOM),
]

TEST_DIR = WORK_DIR / f"test"
rlm_variants = [    
    (d.name.replace("index_", ""), d)
    for d in sorted(WORK_DIR.glob(f"index_rlm_{TOKENIZER}_*_alpha_*"))
    if any(d.glob("segments*"))
]

cg_variants = [
    (d.name.replace("index_", ""), d)
    for d in sorted(WORK_DIR.glob(f"index_cg_*{TOKENIZER}_*alpha_*"))
    if any(d.glob("segments*"))
]

combined_variants = [
    (d.name.replace("index_", ""), d)
    for d in sorted(WORK_DIR.glob(f"index_combined_{TOKENIZER}_*"))
    if any(d.glob("segments*"))
]

if rlm_variants:
    print(f"\nAuto-discovered {len(rlm_variants)} RLM alpha variant(s):")
    for name, path in rlm_variants:
        print(f"  {name} -> {path}")

if cg_variants:
    print(f"\nAuto-discovered {len(cg_variants)} CG (click-graph) alpha variant(s):")
    for name, path in cg_variants:
        print(f"  {name} -> {path}")

if combined_variants:
    print(f"\nAuto-discovered {len(combined_variants)} combined (RLM+CG) variant(s):")
    for name, path in combined_variants:
        print(f"  {name} -> {path}")

active_debug = [(n, p) for n, p in debug_variants if p is not None and any(Path(p).glob("segments*"))]
if active_debug:
    print(f"\n[DEBUG] {len(active_debug)} debug variant(s) found (pipeline verification only):")
    for name, path in active_debug:
        print(f"  {name} -> {path}")

all_variants = fixed_variants + debug_variants + rlm_variants + cg_variants + combined_variants
variants_to_run = []
for variant_name, variant_index_dir in all_variants:
    if variant_index_dir is None:
        print(f"\n[{variant_name.upper()}] Skipped (set to None).")
        continue
    if not any(Path(variant_index_dir).glob("segments*")):
        print(f"\n[{variant_name.upper()}] Skipped — index not found at {variant_index_dir}.")
        continue
    output_path = RESULTS_DIR / f"results_test_{variant_name}.tsv"
    variants_to_run.append((variant_name, str(variant_index_dir), output_path))

print(f"\nRunning retrieval for {len(variants_to_run)} variant(s) "
      f"with {RETRIEVAL_WORKERS} parallel worker(s).")

# Serialise topics to a plain dict so it can be pickled across process boundary
topics_dict = topics_test.to_dict(orient="list")

worker_args = [
    (name, idx_dir, str(out_path), topics_dict, str(WORK_DIR),
     BM25_K1, BM25_B, BATCH_SIZE, NUM_RESULTS)
    for name, idx_dir, out_path in variants_to_run
]

if RETRIEVAL_WORKERS == 1:
    # Single-process mode — easier to debug
    for result_msg in (_retrieval_worker(a) for a in worker_args):
        print(result_msg)
else:
    # Use 'spawn' instead of 'fork' so each worker gets a clean process with
    # its own JVM. Forking an already-started JVM causes workers to run
    # serially instead of in parallel.
    ctx = __import__("multiprocessing").get_context("spawn")
    with ctx.Pool(processes=RETRIEVAL_WORKERS) as pool:
        for result_msg in pool.imap_unordered(_retrieval_worker, worker_args):
            print(result_msg)
 
# =============================================================================
# SUMMARY
# =============================================================================
 
result_files = sorted(WORK_DIR.glob("results_test_*.tsv"))
for path in result_files:
    df = pd.read_csv(path, sep="\t")
    print(f"\n[{path.stem}] {len(df):,} result rows across {df['qid'].nunique():,} queries")
    print(df.head(3))
 
print("\nAll done.")
