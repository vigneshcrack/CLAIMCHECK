"""
Contract tests for Lane 3 — run offline with tiny fake models, so they check the plumbing
(FAISS index, id mapping, ordering, rerank filtering) without downloading anything.

    python -m pytest tests/ -v
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import sbert_faiss  # noqa: E402
import reranker  # noqa: E402


def _words(t):
    return [w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in t).split() if len(w) > 2]


class FakeEncoder:
    """Hashed bag-of-words vectors: similar wording -> similar vectors. Stand-in for SBERT."""
    dim = 256

    def encode(self, texts, normalize_embeddings=True, **kw):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in _words(t):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1
        if normalize_embeddings:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


class FakeCrossEncoder:
    """Scores (claim, evidence) by word overlap. Stand-in for the ms-marco cross-encoder."""

    def predict(self, pairs, **kw):
        return np.array([len(set(_words(a)) & set(_words(b))) - 3.0 for a, b in pairs], dtype="float32")


@pytest.fixture(autouse=True)
def fake_models(tmp_path, monkeypatch):
    monkeypatch.setattr(sbert_faiss, "INDEX_DIR", tmp_path)       # never touch the real indexes/ folder
    monkeypatch.setattr(sbert_faiss, "get_encoder", lambda model_name=None: FakeEncoder())
    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: FakeCrossEncoder())
    sbert_faiss._CACHE.clear()
    yield
    sbert_faiss._CACHE.clear()


CLAIM = "Tilda Swinton is only Scottish."


def test_retrieve_contract():
    out = sbert_faiss.retrieve(CLAIM, 5)
    corpus_ids = set(sbert_faiss.load_corpus()["evidence_id"])
    assert isinstance(out, list) and len(out) == 5
    assert all(isinstance(e, str) and e in corpus_ids for e in out)
    assert len(set(out)) == 5


def test_scores_sorted_and_text_clean():
    hits = sbert_faiss.retrieve_with_scores(CLAIM, 10)
    scores = [s for _, s, _ in hits]
    assert scores == sorted(scores, reverse=True)
    assert all("-LRB-" not in t and "-RRB-" not in t for _, _, t in hits)


def test_k_larger_than_corpus_is_capped():
    n = len(sbert_faiss.load_corpus())
    assert len(sbert_faiss.retrieve(CLAIM, n + 100)) == n


def test_batch_matches_single():
    claims = [CLAIM, "Telemundo is a Spanish-language television network."]
    assert sbert_faiss.retrieve_batch(claims, 5) == [sbert_faiss.retrieve(c, 5) for c in claims]


def test_index_saved_and_reloaded(tmp_path):
    first = sbert_faiss.retrieve(CLAIM, 5)
    assert (tmp_path / "sbert_faiss.index").exists() and (tmp_path / "sbert_meta.json").exists()
    sbert_faiss._CACHE.clear()
    assert sbert_faiss.retrieve(CLAIM, 5) == first


def test_rerank_contract():
    cands = sbert_faiss.retrieve(CLAIM, 20)
    out = reranker.rerank(CLAIM, cands, 5)
    assert len(out) == 5 and set(out) <= set(cands) and len(set(out)) == 5


def test_rerank_edge_cases():
    assert reranker.rerank(CLAIM, [], 5) == []
    assert reranker.rerank(CLAIM, ["not_a_real_id"], 5) == []
    c = sbert_faiss.retrieve(CLAIM, 3)
    assert sorted(reranker.rerank(CLAIM, c + c, 10)) == sorted(c)   # duplicates dropped, top_n > len ok


def test_rerank_works_on_bm25_candidates():
    out = reranker.search(CLAIM, top_n=5, first_stage="bm25", pool_k=20)
    assert len(out) == 5
    assert {"evidence_id", "text", "rank", "rerank_score", "retrieval_score", "retrieval_rank"} <= set(out[0])
    assert all(0 < d["rerank_score"] < 1 for d in out)


def test_package_import_from_project_root():
    root = str(HERE.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import importlib
    m = importlib.import_module("lane3_retrieval_modern.reranker")
    assert hasattr(m, "rerank") and hasattr(m, "search")
