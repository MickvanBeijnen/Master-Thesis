"""
preprocess.py

Unified preprocessing script for both ORCAS (queries) and MS-ORCAS (documents).
Select the dataset and tokenizer pipeline via the CONFIGURATION section below.

DATASET options:
    "orcas"   — Preprocesses ORCAS query file.
                Input format:  qid \t query \t docid \t url
                Output format: qid \t query \t docid

    "msorcas" — Preprocesses MS-ORCAS document file.
                Input format:  docid \t url \t title \t body
                Output format: docid \t title \t body
                Supports checkpointing via RESUME_FROM_INPUT_LINE.

TOKENIZER_MODE options:
    "anserini"        — Anserini's DefaultEnglishAnalyzer end-to-end:
                        StandardTokenizer (UAX #29) + lowercase + Lucene
                        English stopwords + Porter stemmer. Standard baseline.

    "symbol"   — Java PatternTokenizer keeping compound terms like c++, covid-19.
    "alphanum" — Java PatternTokenizer with alphanumeric-only regex (no symbols).
                        merges tokens connected by + # = @ * % / _ & ^ ~ < > -
                        so that e.g. "covid-19", "e-mail", "c++" are kept as
                        single tokens. Requires CustomRegexAnalyzer.class.
"""

import re
import unicodedata
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose one of: "anserini", "symbol" or "alphanum"
TOKENIZER_MODE = "anserini"

# Input paths
ORCAS_INPUT_PATH   = Path("../Datasets/orcas.tsv")
MSORCAS_INPUT_PATH = Path("../Datasets/msorcas.tsv")

# Output paths (auto-named by dataset + tokenizer mode)
ORCAS_OUTPUT_PATH   = Path(f"../Datasets/orcas_preprocessed_{TOKENIZER_MODE}.tsv")
MSORCAS_OUTPUT_PATH = Path(f"../Datasets/msorcas_preprocessed_{TOKENIZER_MODE}.tsv")

# MS-ORCAS checkpointing: number of input lines already processed in a
# previous run. Set to 0 to start from the beginning.
# Set to the number of lines in the existing output file to resume.
RESUME_FROM_INPUT_LINE = 0

PROGRESS_EVERY = 100_000   # for msorcas
ORCAS_PROGRESS_EVERY = 500_000

# Path to Anserini's fat-jar. Required for all modes.
# Leave as None to auto-detect from pyserini's installed location.
# Set manually if auto-detection fails, e.g.:
#   ANSERINI_JAR_PATH = r"C:\path\to\venv\Lib\site-packages\pyserini\resources\jars\anserini-1.1.1-fatjar.jar"
ANSERINI_JAR_PATH = None

# Directory containing CustomRegexAnalyzer.class.
# Only needed for TOKENIZER_MODE = "symbol" or "alphanum".
# Compile with: javac -proc:none -cp "anserini-*-fatjar.jar" CustomRegexAnalyzer.java
CUSTOM_ANALYZER_DIR = Path(".")

# =============================================================================
# JVM SETUP — must happen before any jnius import
# =============================================================================

import jnius_config

if not jnius_config.vm_running:
    jar_path = ANSERINI_JAR_PATH

    if jar_path is None:
        try:
            import pyserini
            pyserini_dir = Path(pyserini.__file__).parent
            candidates = list(
                (pyserini_dir / "resources" / "jars").glob("anserini-*-fatjar.jar")
            )
            if candidates:
                jar_path = str(candidates[0])
        except ImportError:
            pass

    if jar_path is None:
        raise RuntimeError(
            "Could not locate the Anserini fat-jar automatically. "
            "Set ANSERINI_JAR_PATH at the top of this script."
        )

    print(f"Adding Anserini fat-jar to classpath: {jar_path}")
    jnius_config.add_classpath(jar_path)

    if TOKENIZER_MODE in ("symbol", "alphanum"):
        jnius_config.add_classpath(str(CUSTOM_ANALYZER_DIR.resolve()))

from jnius import autoclass

JAnalyzerUtils          = autoclass("io.anserini.analysis.AnalyzerUtils")
JDefaultEnglishAnalyzer = autoclass("io.anserini.analysis.DefaultEnglishAnalyzer")

_anserini_analyzer = JDefaultEnglishAnalyzer.newDefaultInstance()


def _anserini_analyze(text: str) -> list[str]:
    """Run text through Anserini's full DefaultEnglishAnalyzer pipeline."""
    if not text:
        return []
    return [str(t) for t in JAnalyzerUtils.analyze(_anserini_analyzer, text)]


# Java-based custom analyzers: load only when needed
_java_custom_analyzer = None
if TOKENIZER_MODE in ("symbol", "alphanum"):
    JCustomRegexAnalyzer = autoclass("CustomRegexAnalyzer")
    _java_custom_analyzer = JCustomRegexAnalyzer(TOKENIZER_MODE)
    print(f"CustomRegexAnalyzer loaded in {TOKENIZER_MODE} mode.")


def _java_custom_analyze(text: str) -> list[str]:
    if not text:
        return []
    return [str(t) for t in JAnalyzerUtils.analyze(_java_custom_analyzer, text)]


_anserini_stopword_set = _anserini_analyzer.getStopwordSet()
LUCENE_STOPWORDS: set[str] = set()
_it = _anserini_stopword_set.iterator()
while _it.hasNext():
    _raw = _it.next()
    _word = "".join(_raw) if hasattr(_raw, "__iter__") and not isinstance(_raw, str) else str(_raw)
    LUCENE_STOPWORDS.add(_word)

print(f"Loaded {len(LUCENE_STOPWORDS)} stopwords from Anserini's default list.")

# =============================================================================
# DISPATCH
# =============================================================================

if TOKENIZER_MODE == "anserini":
    def preprocess_text(text: str) -> list[str]:
        return _anserini_analyze(text)

elif TOKENIZER_MODE in ("symbol", "alphanum"):
    def preprocess_text(text: str) -> list[str]:
        return _java_custom_analyze(text)

else:
    raise ValueError(
        f"Unknown TOKENIZER_MODE: {TOKENIZER_MODE!r}. "
        f"Choose 'anserini', 'symbol', or'alphanum'."
    )


def tokens_to_text(tokens: list[str]) -> str:
    return " ".join(tokens)


# =============================================================================
# ORCAS PROCESSING
# =============================================================================

def process_orcas(input_path: Path, output_path: Path) -> None:
    print(f"Input file    : {input_path}")
    print(f"Output file   : {output_path}")

    processed_rows = 0
    bad_lines = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as infile, \
         output_path.open("w", encoding="utf-8", errors="replace", newline="") as outfile:

        for line_number, line in enumerate(infile, start=1):
            line = line.rstrip("\n")
            # ORCAS format: qid \t query \t docid \t url
            parts = line.split("\t", 3)
            if len(parts) < 4:
                bad_lines += 1
                continue

            qid, query, docid, url = parts
            query_tokens = preprocess_text(query)
            if not query_tokens:
                bad_lines += 1  # query became empty after preprocessing
                continue
            outfile.write(f"{qid}\t{tokens_to_text(query_tokens)}\t{docid}\n")
            processed_rows += 1

            if processed_rows % ORCAS_PROGRESS_EVERY == 0:
                print(
                    f"Processed rows: {processed_rows:,} | "
                    f"Current input line: {line_number:,} | "
                    f"Bad lines: {bad_lines:,}"
                )

    print("\nDone.")
    print(f"Processed rows : {processed_rows:,}")
    print(f"Bad lines      : {bad_lines:,}")
    print(f"Output file    : {output_path}")


# =============================================================================
# MS-ORCAS PROCESSING  (with checkpointing)
# =============================================================================

def process_msorcas(
    input_path: Path,
    output_path: Path,
    resume_from: int,
) -> None:
    print(f"Input file            : {input_path}")
    print(f"Output file           : {output_path}")
    print(f"Resume from input line: {resume_from:,}")

    processed_now = 0
    skipped = 0
    bad_lines = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as infile, \
         output_path.open("a", encoding="utf-8", errors="replace", newline="") as outfile:

        for line_number, line in enumerate(infile, start=1):
            if line_number <= resume_from:
                skipped += 1
                continue

            line = line.rstrip("\n")
            # MS-ORCAS format: docid \t url \t title \t body
            parts = line.split("\t", 3)
            if len(parts) < 4:
                bad_lines += 1
                continue

            docid, url, title, body = parts
            title_text = tokens_to_text(preprocess_text(title or ""))
            body_text  = tokens_to_text(preprocess_text(body  or ""))

            outfile.write(f"{docid}\t{title_text}\t{body_text}\n")
            processed_now += 1

            if processed_now % PROGRESS_EVERY == 0:
                print(
                    f"Processed this run: {processed_now:,} | "
                    f"Skipped old rows: {skipped:,} | "
                    f"Current input line: {line_number:,} | "
                    f"Bad lines: {bad_lines:,}"
                )

    print("\nDone.")
    print(f"Skipped previous rows: {skipped:,}")
    print(f"Processed this run   : {processed_now:,}")
    print(f"Bad lines skipped    : {bad_lines:,}")
    print(f"Output file          : {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print(f"Tokenizer mode: {TOKENIZER_MODE}")
    print()

    print("=" * 60)
    print("ORCAS")
    print("=" * 60)
    if not ORCAS_INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {ORCAS_INPUT_PATH}")
    process_orcas(ORCAS_INPUT_PATH, ORCAS_OUTPUT_PATH)

    print()

    print("=" * 60)
    print("MS-ORCAS")
    print("=" * 60)
    if not MSORCAS_INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {MSORCAS_INPUT_PATH}")
    process_msorcas(MSORCAS_INPUT_PATH, MSORCAS_OUTPUT_PATH, RESUME_FROM_INPUT_LINE)


if __name__ == "__main__":
    main()
