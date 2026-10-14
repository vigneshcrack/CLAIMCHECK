"""
build_index.py — builds and saves the Lane 2 BM25 and TF-IDF indexes over Lane 1's evidence corpus.

    python build_index.py

Reads  ../lane1_data_eval/data/processed/evidence_corpus.csv   (evidence_id, text)
Writes indexes/bm25.pkl, indexes/tfidf.pkl
(retrieval.py also builds these automatically on first use if they are missing.)
"""
import time

from retrieval import CORPUS_CSV, INDEX_DIR, K1, B, build_index, load_corpus

if __name__ == "__main__":
    n = len(load_corpus(CORPUS_CSV))
    for method in ("bm25", "tfidf"):
        t0 = time.time()
        build_index(method, CORPUS_CSV, save=True)
        print(f"{method:5s}: indexed {n} evidence sentences in {time.time() - t0:.2f}s -> {INDEX_DIR / (method + '.pkl')}")
    print(f"BM25 parameters: k1={K1}, b={B}")
