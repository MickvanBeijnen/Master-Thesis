"""
sort_weights.py

Sorts a weights TSV into the same order as the Lucene index traversal
and prepends the integer traversal index to each row, using an external
merge sort so that only one chunk fits in memory at a time.

Output columns: traversal_idx, doc_id, term, probability

Supports two input formats:
  4 columns: traversal_idx, doc_id, term, probability  (clickgraph format)
  3 columns: doc_id, term, probability                  (RLM format)
The column count is detected automatically per row.

Usage:
    python sort_weights.py <index_dir> <weights_tsv> <sorted_output_tsv> [chunk_size]

    index_dir          Path to the Anserini/Lucene index used to determine
                        traversal order (e.g. anserini_index_num)
    weights_tsv         Input weights file with columns:
                        (traversal_idx placeholder, doc_id, term, probability)
                        The first column is ignored/overwritten; doc_id is
                        looked up against the index's real traversal order.
                        A header row is auto-detected (first field non-numeric)
                        and skipped if present; works with or without one.
    sorted_output_tsv   Output path for the sorted, traversal-indexed TSV
    chunk_size          Optional: rows per sort chunk (default: 5,000,000).
                        Reduce if you hit memory limits.

If no arguments are given, falls back to the original hardcoded RLM paths
for backward compatibility with earlier interactive usage.
"""

import argparse
import csv
import heapq
import os
import sys
import tempfile
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher

# ── Fallback defaults (used only if no CLI args are given) ───────────────────
TOKENIZER = "anserini"   # "anserini", "symbol" or "alphanum"
METHOD = "lavrenko"      # "craswell" or "lavrenko"

WORK_DIR   = Path("..")
INDEX_DIR = WORK_DIR / f"anserini_index_{TOKENIZER}"
CHUNK_SIZE = 5_000_000

if METHOD == "craswell":
    WEIGHTS_PATH = WORK_DIR / f"clickgraph_weights_{TOKENIZER}_raw.tsv"
    SORTED_PATH = WORK_DIR / f"clickgraph_weights_{TOKENIZER}_sorted.tsv"
elif METHOD == "lavrenko":
    WEIGHTS_PATH = WORK_DIR / f"rlm_weights_{TOKENIZER}_raw.tsv"
    SORTED_PATH = WORK_DIR / f"rlm_weights_{TOKENIZER}_sorted.tsv"


def parse_args():
    p = argparse.ArgumentParser(
        description="External merge-sort a weights TSV into Lucene traversal order."
    )
    p.add_argument("index_dir", nargs="?", default=INDEX_DIR,
                    help="Path to the Anserini/Lucene index")
    p.add_argument("weights_tsv", nargs="?", default=WEIGHTS_PATH,
                    help="Input weights TSV (doc_id, term, probability; first column ignored)")
    p.add_argument("sorted_output_tsv", nargs="?", default=SORTED_PATH,
                    help="Output path for sorted, traversal-indexed TSV")
    p.add_argument("chunk_size", nargs="?", type=int, default=CHUNK_SIZE,
                    help="Rows per sort chunk (default: 5,000,000)")
    p.add_argument("--tmp-dir", default=None,
                    help="Directory for temp chunk files (default: alongside sorted_output_tsv)")
    return p.parse_args()


def parse_row(row: list) -> tuple | None:
    """
    Parse a data row that may have 3 or 4 columns:
      4 columns: traversal_idx, doc_id, term, probability  (clickgraph format)
      3 columns: doc_id, term, probability                  (RLM format)
    Returns (doc_id, term, probability) or None if the row is malformed.
    """
    if len(row) == 4:
        return row[1].strip(), row[2].strip(), row[3].strip()
    elif len(row) == 3:
        return row[0].strip(), row[1].strip(), row[2].strip()
    return None


def main():
    args = parse_args()

    index_dir   = Path(args.index_dir)
    weights_path = Path(args.weights_tsv)
    sorted_path  = Path(args.sorted_output_tsv)
    chunk_size   = args.chunk_size

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else sorted_path.parent / "sort_tmp"

    print(f"Index dir    : {index_dir}")
    print(f"Weights TSV  : {weights_path}")
    print(f"Sorted output: {sorted_path}")
    print(f"Chunk size   : {chunk_size:,}")
    print(f"Temp dir     : {tmp_dir}")

    # ── Step 1 — build doc_id -> traversal_idx mapping ───────────────────────
    print("Reading index traversal order...")
    searcher = LuceneSearcher(str(index_dir))
    order = {}
    for i in range(searcher.num_docs):
        doc = searcher.doc(i)
        if doc:
            order[doc.docid()] = i
    print(f"Got traversal order for {len(order):,} documents.")

    # ── Step 2 — split weights file into sorted chunks ───────────────────────
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunk_files = []
    chunk = []
    total_rows = 0
    skipped_no_idx = 0

    print(f"Splitting into chunks of {chunk_size:,} rows...")

    with open(weights_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        first_row = next(reader, None)
        if first_row is None:
            print("Warning: input file is empty.")
        else:
            # Auto-detect header: first field of a header row is non-numeric
            try:
                int(first_row[0].strip())
                is_header = False
            except (ValueError, IndexError):
                # Could still be a 3-column RLM header (doc_id is non-numeric)
                # Check if it looks like a doc ID (starts with D followed by digits)
                # or a column name
                first_field = first_row[0].strip()
                is_header = not (first_field.startswith("D") and
                                 first_field[1:].isdigit()) if first_field else True

            if is_header:
                print(f"  Detected header row, skipping: {first_row}")
            else:
                row_data = parse_row(first_row)
                if row_data:
                    doc_id, term, prob = row_data
                    idx = order.get(doc_id)
                    if idx is None:
                        skipped_no_idx += 1
                    else:
                        chunk.append((idx, doc_id, term, prob))
                        total_rows += 1

        for row in reader:
            row_data = parse_row(row)
            if not row_data:
                continue
            doc_id, term, prob = row_data
            idx = order.get(doc_id)
            if idx is None:
                skipped_no_idx += 1
                continue

            chunk.append((idx, doc_id, term, prob))
            total_rows += 1

            if len(chunk) >= chunk_size:
                chunk.sort(key=lambda x: x[0])
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".tsv", dir=tmp_dir,
                    delete=False, encoding="utf-8"
                )
                w = csv.writer(tmp, delimiter="\t")
                w.writerows(chunk)
                tmp.close()
                chunk_files.append(tmp.name)
                print(f"  Written chunk {len(chunk_files)} ({total_rows:,} rows so far)")
                chunk.clear()

    # Write final partial chunk
    if chunk:
        chunk.sort(key=lambda x: x[0])
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", dir=tmp_dir,
            delete=False, encoding="utf-8"
        )
        w = csv.writer(tmp, delimiter="\t")
        w.writerows(chunk)
        tmp.close()
        chunk_files.append(tmp.name)
        print(f"  Written final chunk {len(chunk_files)} ({total_rows:,} rows total)")
        chunk.clear()

    print(f"Split into {len(chunk_files)} chunks covering {total_rows:,} rows "
          f"({skipped_no_idx:,} rows skipped: doc_id not found in index).")

    # ── Step 3 — k-way merge the sorted chunks into the output file ──────────
    print(f"Merging chunks into {sorted_path}...")

    def open_chunk(path):
        """Open a chunk file and yield (traversal_idx, row_tuple) for heapq."""
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                yield (int(row[0]), row)

    handles = [open_chunk(p) for p in chunk_files]
    written = 0

    sorted_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sorted_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["traversal_idx", "doc_id", "term", "probability"])
        for _idx, row in heapq.merge(*handles, key=lambda x: x[0]):
            writer.writerow(row)
            written += 1
            if written % 5_000_000 == 0:
                print(f"  Merged {written:,} rows...")

    print(f"Done. Wrote {written:,} rows to {sorted_path}.")

    # ── Step 4 — clean up temp files ──────────────────────────────────────────
    print("Cleaning up temp files...")
    for p in chunk_files:
        os.unlink(p)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # not empty / shared dir — leave it
    print("All done.")


if __name__ == "__main__":
    main()
