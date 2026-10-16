"""Unified Lane 2/3 retrieval -> Lane 4 NLI -> optional experimental claim aggregation.

All five methods return the same evidence schema. No cross-method score thresholds.
Use verify_retrieved_claim() in Lane 5; assess aggregate strategy on dev first.
"""
from __future__ import annotations
import math
from .claim_verdict import verify_claim_from_evidence

METHODS = ('bm25', 'tfidf', 'sbert', 'sbert_rerank', 'bm25_rerank')


def retrieve_ranked_evidence(claim: str, *, method: str = 'bm25', k: int = 5) -> list[dict]:
    if method not in METHODS:
        raise ValueError(f'Unknown retrieval method {method!r}; expected one of {METHODS}')
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError('claim must be a non-empty string')
    if type(k) is not int or k < 1:
        raise ValueError('k must be a positive integer')

    if method in ('bm25', 'tfidf'):
        from lane2_retrieval_traditional.retrieval import retrieve_with_scores
        hits = retrieve_with_scores(claim, k=k, method=method)
        rows = [(eid, score, text, None) for eid, score, text in hits]
    elif method == 'sbert':
        from lane3_retrieval_modern.sbert_faiss import retrieve_with_scores
        hits = retrieve_with_scores(claim, k=k)
        rows = [(eid, score, text, None) for eid, score, text in hits]
    else:
        from lane3_retrieval_modern.reranker import search
        stage = 'sbert' if method == 'sbert_rerank' else 'bm25'
        hits = search(claim, top_n=k, first_stage=stage)
        rows = [(hit['evidence_id'], hit['rerank_score'], hit['text'],
                 hit.get('retrieval_score')) for hit in hits]

    out, seen = [], set()
    for eid, score, text, stage_score in rows:
        eid = str(eid)
        score = float(score)
        if eid in seen or not math.isfinite(score) or not str(text).strip():
            continue
        # Lane 2's zero/negative lexical scores are not useful. Semantic/rerank scores
        # do not have the same interpretation; never filter them by positivity.
        if method in ('bm25', 'tfidf') and score <= 0:
            continue
        seen.add(eid)
        item = dict(evidence_id=eid, text=str(text), retrieval_score=score,
                    retrieval_method=method, rank=len(out) + 1)
        if stage_score is not None:
            item['first_stage_score'] = float(stage_score)
        out.append(item)
    return out[:k]


def verify_retrieved_claim(claim: str, *, method: str = 'bm25', k: int = 5,
                           strategy: str = 'conflict_abstain', batch_size: int = 8) -> dict:
    """Return {'claim','method','top_k','evidence','verdict'} with aligned evidence IDs.

    Claim verdict is EXPERIMENTAL and must be evaluated before being represented as
    a validated fact-check. Individual NLI predictions remain directly inspectable.
    """
    evidence = retrieve_ranked_evidence(claim, method=method, k=k)
    out = verify_claim_from_evidence(claim, evidence,
                                     strategy=strategy, batch_size=batch_size)
    return {'method': method, 'top_k': k, **out}
