# Lightweight Term Weighting Strategies for First-Stage Passage Retrieval

Master's thesis codebase implementing custom term-frequency weighting for first-stage passage retrieval using Lucene/Anserini indices, inspired by Dai et al. (2020).

## Requirements

Python 3.10 and Java 21 are required.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Export the Anserini fat-jar path (required for preprocessing and index building):

```bash
export FATJAR=$(python3 -c "import pyserini; from pathlib import Path; print(next((Path(pyserini.__file__).parent / 'resources' / 'jars').glob('anserini-*-fatjar.jar')))")
```

Compile the Java files before use:

```bash
javac -proc:none -cp "$FATJAR" Preprocessing/CustomRegexAnalyzer.java
javac -proc:none -cp "$FATJAR" Term_Weighting_and_Index_Building/ReweightIndex.java
```

---

### Step 0 — Obtain Datasets

Before running the preprocessing pipeline, create a `Datasets/` folder in the root of the repository and place the raw MS MARCO and ORCAS dataset files inside it:

```
Datasets/
├── msmarco.tsv
└── orcas.tsv
```

Download links and version details can be found in the [Dataset](#dataset) section.

### Step 1 — Preprocessing

First, generate the MS-ORCAS dataset:

```bash
python Preprocessing/msorcas_generator.py
```

Then preprocess both the MS-ORCAS document corpus and the ORCAS query file. Set `TOKENIZER_MODE` at the top of the script to one of `"anserini"`, `"symbol_anserini"`, or `"alphanum_anserini"`. `CustomRegexAnalyzer.class` must be present in the same directory as the script when using `symbol_anserini` or `alphanum_anserini` modes.

```bash
python Preprocessing/msorcas_preprocesser.py
```

---

### Step 2 — Build baseline index

The baseline index is built automatically by `bm25.py` if it does not already exist. Simply run:

```bash
python Retrieval_and_Evaluation/bm25.py
```

This will build the index, run retrieval over all discovered variants, and write results to the results directory.

### Step 3 — Generate term weights

Configure paths and hyper-parameters at the top of each script, then run:

**Lavrenko (Relevance-Based Language Models):**

```bash
python Term_Weighting_and_Index_Building/lavrenko_weights.py
```

**Craswell (Click-Graph PPR):**

```bash
python Term_Weighting_and_Index_Building/craswell_weights.py
```

### Step 4 — Sort weights into index traversal order

```bash
python Term_Weighting_and_Index_Building/sort_weights.py \
    <index_dir> \
    <weights_raw.tsv> \
    <weights_sorted.tsv>
```

### Step 5 — Build reweighted index

Reweighted indices are built using `ReweightIndex.java`. All modes use log-scaling to map raw weights into a fixed integer range `[1, max_val+1]`. Missing weights fall back to the original TF scaled by the corresponding coefficient.

**Single-method interpolation** (`rlm` / `cg`):
```
new_tf = α·tf + (1−α)·weight
```
```bash
java -cp ".:$FATJAR" -Xmx8g ReweightIndex \
    <input_index> <output_dir> <tokenizer> rlm <max_val> <rlm_weights.tsv> <alpha>
```
Replace `rlm` with `cg` and provide the click-graph weights file to use click-graph weights instead. Append `_multi` to the mode name and pass a comma-separated list of alpha values to build all variants in a single pass.

**Combined interpolation** (`combined` / `combined_multi`):
```
new_tf = α·tf + β·rlm_weight + (1−α−β)·cg_weight    (requires α + β ≤ 1)
```
```bash
java -cp ".:$FATJAR" -Xmx8g ReweightIndex \
    <input_index> <output_dir> <tokenizer> combined_multi <max_val_rlm> <max_val_cg> \
    <rlm_weights.tsv> <cg_weights.tsv> <alpha1,alpha2,...> <beta1,beta2,...>
```
Note that `combined_multi` takes separate `max_val` parameters for RLM and click-graph weights. Invalid pairs where α + β > 1 are skipped automatically.

### Step 6 — Retrieval and evaluation

```bash
python Retrieval_and_Evaluation/bm25.py
python Retrieval_and_Evaluation/evaluate.py
```

Configure paths at the top of each script. `bm25.py` auto-discovers reweighted index variants by globbing for `index_rlm_*`, `index_cg_*`, and `index_combined_*` in the work directory.

## Dataset

This codebase uses two datasets from the [MS MARCO](https://microsoft.github.io/msmarco/) project:

- **MS MARCO Document Ranking v1** — the document corpus (`msmarco-docs.tsv`), available on the [MS MARCO Datasets page](https://microsoft.github.io/msmarco/Datasets.html)
- **ORCAS** — the full click-based query dataset (~18 million entries mapping to 1.4 million of the MS MARCO documents), available at [microsoft.github.io/msmarco/ORCAS.html](https://microsoft.github.io/msmarco/ORCAS.html)

From these two datasets, `msorcas_generator.py` generates the intersecting dataset we refer to as **MS ORCAS** — containing only documents from MS MARCO that appear in ORCAS. The preprocessed corpus and index are not included in this repository due to size.

## References

Dai, Z., & Callan, J. (2020). Context-Aware Term Weighting For First Stage Passage Retrieval. *SIGIR 2020*.

Lavrenko, V., & Croft, W. B. (2001). Relevance-Based Language Models. *SIGIR 2001*.

Craswell, N., Szummer, M. (2007). Random Walks on the Click Graph. *SIGIR 2007*.
