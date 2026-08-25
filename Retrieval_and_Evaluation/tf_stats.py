from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher
import numpy as np

WORK_DIR   = Path("..")
INDEX_DIR = WORK_DIR / f"anserini_index_{TOKENIZER}"
reader  = LuceneIndexReader(INDEX_DIR)
searcher = LuceneSearcher(INDEX_DIR)

sample_tfs = []
for lucene_id in range(0, 1_000_000_000):
    doc = searcher.doc(lucene_id)
    if doc is None:
        continue
    external_id = doc.docid()
    tf_vector = reader.get_document_vector(external_id)
    if tf_vector:
        sample_tfs.extend(tf_vector.values())

print(f"Collected {len(sample_tfs):,} TF values")
print(np.percentile(sample_tfs, [50, 75, 90, 95, 99, 99.9]))
