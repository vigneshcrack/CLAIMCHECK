"""
reranker.py — Lane 3: cross-encoder reranking for ClaimCheck.

Shared interface contract:

    def rerank(claim: str, candidates: list[str], top_n: int) -> list[str]
        Re-order/filter candidate evidence_ids, best-first, down to top_n.

Works on candidates from ANY first-stage retriever (Lane 3 SBERT or Lane 2 BM25), because it
only needs evidence_ids from Lane 1's corpus.

Extras:
    rerank_with_scores(claim, candidates, top_n) -> [(evidence_id, rerank_score, readable_text), ...]
    search(claim, top_n=5, first_stage="sbert", pool_k=POOL_K)
        one-call pipeline for the app: first-stage retrieve pool_k -> rerank -> top_n
        -> [{"evidence_id", "text", "rerank_score", "retrieval_score", "retrieval_rank", "rank"}, ...]

Design:
  * Checkpoint: cross-encoder/ms-marco-MiniLM-L-6-v2. A cross-encoder reads (claim, evidence) together
    with full attention between the two, so it can weigh the one or two words that separate
    FEVER-Symmetric "twin" sentences, which a bi-encoder (one vector per sentence) or BM25 blurs.
  * Raw output is a relevance logit; rerank_score = sigmoid(logit) in (0, 1) for display only
    (it is a relevance score, NOT the probability that the claim is true — that is Lane 4's job).
"""

from __future__ import annotations

import os

import numpy as np

try:                                   # imported as a package from the project root (e.g. by the app)
    from .sbert_faiss import CORPUS_CSV, clean_text, load_corpus
    from . import sbert_faiss as _sbert
except ImportError:                    # run from inside this folder
    from sbert_faiss import CORPUS_CSV, clean_text, load_corpus
    import sbert_faiss as _sbert

CROSS_ENCODER_MODEL = os.environ.get("CLAIMCHECK_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
POOL_K = 50          # first-stage candidates passed to the cross-encoder (chosen on dev in evaluate_lane3.py)
MAX_LENGTH = 256     # claim + evidence tokens; FEVER sentences are short, so nothing is truncated in practice

_MODEL = None
_TEXT: dict | None = None


def get_cross_encoder():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder
        _MODEL = CrossEncoder(CROSS_ENCODER_MODEL, max_length=MAX_LENGTH)
    return _MODEL


def _texts() -> dict:
    global _TEXT
    if _TEXT is None:
        df = load_corpus(CORPUS_CSV)
        _TEXT = {e: clean_text(t) for e, t in zip(df["evidence_id"], df["text"])}
    return _TEXT


def _dedupe_known(candidates: list[str]) -> list[str]:
    """Keep first occurrence, drop ids that are not in Lane 1's corpus."""
    seen, out, text = set(), [], _texts()
    for c in candidates:
        c = str(c)
        if c in text and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def score_pairs(claim: str, evidence_ids: list[str]) -> np.ndarray:
    if not evidence_ids:
        return np.zeros(0, dtype="float32")
    text = _texts()
    pairs = [(clean_text(claim), text[e]) for e in evidence_ids]
    return np.asarray(get_cross_encoder().predict(pairs, show_progress_bar=False), dtype="float32").ravel()


def rerank_with_scores(claim: str, candidates: list[str], top_n: int) -> list[tuple[str, float, str]]:
    cands = _dedupe_known(candidates)
    if not cands or top_n <= 0:
        return []
    logits = score_pairs(claim, cands)
    order = np.argsort(-logits, kind="stable")[:top_n]       # stable: ties keep first-stage order
    text = _texts()
    return [(cands[i], float(1 / (1 + np.exp(-logits[i]))), text[cands[i]]) for i in order]


def rerank(claim: str, candidates: list[str], top_n: int) -> list[str]:
    """Re-order/filter candidate evidence_ids, best-first, down to top_n."""
    return [eid for eid, _, _ in rerank_with_scores(claim, candidates, top_n)]


def search(claim: str, top_n: int = 5, first_stage: str = "sbert", pool_k: int = POOL_K) -> list[dict]:
    """Full Lane 3 pipeline for the app: first-stage retrieval -> cross-encoder rerank -> top_n."""
    if first_stage == "sbert":
        first = _sbert.retrieve_with_scores(claim, pool_k)
    elif first_stage == "bm25":
        import sys
        from pathlib import Path
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from lane2_retrieval_traditional.retrieval import retrieve_with_scores as bm25_rws
        first = bm25_rws(claim, pool_k, method="bm25")
    else:
        raise ValueError("first_stage must be 'sbert' or 'bm25'")
    first_score = {e: s for e, s, _ in first}
    first_rank = {e: r for r, (e, _, _) in enumerate(first, 1)}
    out = []
    for rank, (eid, rs, txt) in enumerate(rerank_with_scores(claim, [e for e, _, _ in first], top_n), 1):
        out.append(dict(evidence_id=eid, text=txt, rank=rank, rerank_score=rs,
                        retrieval_score=first_score[eid], retrieval_rank=first_rank[eid]))
    return out
