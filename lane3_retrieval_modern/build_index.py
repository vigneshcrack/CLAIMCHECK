"""
build_index.py — builds and saves the Lane 3 SBERT + FAISS index over Lane 1's evidence corpus.

    python build_index.py

Reads  ../lane1_data_eval/data/processed/evidence_corpus.csv   (evidence_id, text)
Writes indexes/sbert_faiss.index  (FAISS IndexFlatIP over L2-normalised embeddings)
       indexes/sbert_meta.json    (model name, dim, evidence_ids in index order, cleaned texts, corpus fingerprint)
(sbert_faiss.py also builds these automatically on first use, and rebuilds them if the corpus changes.)
The first run downloads the model from Hugging Face (~420 MB for all-mpnet-base-v2).
"""
import time

from sbert_faiss import CORPUS_CSV, INDEX_DIR, SBERT_MODEL, build_index

if __name__ == "__main__":
    t0 = time.time()
    idx = build_index(CORPUS_CSV, SBERT_MODEL, save=True)
    print(f"SBERT model : {SBERT_MODEL}")
    print(f"Embedding   : {idx.dim}-d, L2-normalised, FAISS IndexFlatIP (exact cosine search)")
    print(f"Indexed {len(idx.ids)} evidence sentences in {time.time() - t0:.1f}s -> {INDEX_DIR}")
