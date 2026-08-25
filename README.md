# Custom Term-Frequency Weighting for First-Stage Passage Retrieval

Master's thesis codebase implementing custom term-frequency weighting for first-stage passage retrieval using Lucene/Anserini indices, inspired by Dai et al. (2020).

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Java 21 and the Anserini fat-jar (bundled with Pyserini) are required for indexing, retrieval, and the custom Java analyzer. The fat-jar is auto-detected from the Pyserini installation.

Compile the Java files before use:

```bash
export FATJAR=$(python3 -c "import pyserini; from pathlib import Path; print(next((Path(pyserini.__file__).parent / 'resources' / 'jars').glob('anserini-*-fatjar.jar')))")

javac -proc:none -cp "$FATJAR" Preprocessing/CustomRegexAnalyzer.java
javac -proc:none -cp "$FATJAR" Term_Weighting_and_Index_Building/ReweightIndex.java
```

## Pipeline

### Step 1 — Preprocessing

Preprocess both the MS-ORCAS document corpus and the ORCAS query file. Set `TOKENIZER_MODE` at the top of the script to one of `"anserini"`, `"symbol_anserini"`, or `"alphanum_anserini"`.

```bash
python Preprocessing/preprocess.py
```

`CustomRegexAnalyzer.class` must be in the same directory as the script when using `symbol_anserini` or `alphanum_anserini` modes.

### Step 2 — Build baseline index

```bash
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input <corpus_dir> \
    --index <index_dir> \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 \
    --pretokenized
```

### Step 3 — Generate term weights

**Craswell (Clickgraph PPR):**

```bash
python Term_Weighting_and_Index_Building/craswell_weights.py \
    --orcas       <orcas_preprocessed.tsv> \
    --qrels-train <qrels_train.tsv> \
    --corpus      <corpus.jsonl> \
    --output      <clickgraph_weights_raw.tsv> \
    --tokenizer   anserini \
    --weight-by-tf
```

**Lavrenko (RLM):**

Configure paths and hyperparameters at the top of the script, then:

```bash
python Term_Weighting_and_Index_Building/lavrenko_weights.py
```

### Step 4 — Sort weights into index traversal order

```bash
python Term_Weighting_and_Index_Building/sort_weights.py \
    <index_dir> \
    <weights_raw.tsv> \
    <weights_sorted.tsv>
```

### Step 5 — Build reweighted index

```bash
java -cp ".:$FATJAR" -Xmx8g Term_Weighting_and_Index_Building/ReweightIndex \
    <input_index> \
    <output_dir> \
    <tokenizer> \
    <mode> <max_val> \
    <weights_sorted.tsv> \
    <alpha>
```

Modes: `rlm`, `rlm_multi`, `cg`, `cg_multi`, `combined`, `combined_multi`. See the file header for full argument documentation per mode.

### Step 6 — Retrieval and evaluation

```bash
python Retrieval_and_Evaluation/bm25.py
python Retrieval_and_Evaluation/evaluate.py
```

Configure paths at the top of each script. `bm25.py` auto-discovers reweighted index variants by globbing for `index_rlm_*`, `index_cg_*`, and `index_combined_*` in the work directory.

## Dataset

This codebase was developed using the MS ORCAS dataset. The preprocessed corpus and index are not included in this repository due to size.

## Reference

Dai, Z., & Callan, J. (2020). Context-Aware Term Weighting For First Stage Passage Retrieval. *SIGIR 2020*.
