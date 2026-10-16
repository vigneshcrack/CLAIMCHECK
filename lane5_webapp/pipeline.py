from __future__ import annotations

import math
from functools import lru_cache


LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")

METHOD_LABELS = {
    "bm25": "BM25 — lexical retrieval",
    "tfidf": "TF-IDF — vector-space retrieval",
    "sbert": "SBERT + FAISS — semantic retrieval",
    "sbert_rerank": "SBERT + Cross-Encoder reranking",
    "bm25_rerank": "BM25 + Cross-Encoder reranking",
}


@lru_cache(maxsize=5)
def load_retrieval_index(method: str):
    """Build or reuse the appropriate retrieval index."""

    if method not in METHOD_LABELS:
        raise ValueError(f"Unsupported retrieval method: {method}")

    if method in ("bm25", "tfidf", "bm25_rerank"):
        from lane2_retrieval_traditional.retrieval import get_index

        index_method = (
            "bm25" if method == "bm25_rerank" else method
        )

        return get_index(index_method)

    from lane3_retrieval_modern.sbert_faiss import get_index

    return get_index()


def retrieve_evidence(claim: str, k: int, method: str):
    """Retrieve evidence as (evidence_id, score, text) tuples."""

    if method not in METHOD_LABELS:
        raise ValueError(
            f"Unsupported retrieval method: {method}"
        )

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Claim cannot be empty")

    load_retrieval_index(method)

    # --------------------------------------------------------
    # Traditional retrieval: BM25 and TF-IDF
    # --------------------------------------------------------

    if method in ("bm25", "tfidf"):

        from lane2_retrieval_traditional.retrieval import (
            retrieve_with_scores,
        )

        candidates = retrieve_with_scores(
            claim=claim,
            k=k,
            method=method,
        )

        return [
            (evidence_id, numeric_score, text)
            for evidence_id, score, text in candidates
            if math.isfinite(numeric_score := float(score))
            and numeric_score > 0
        ]

    # --------------------------------------------------------
    # Modern retrieval: SBERT + FAISS
    # --------------------------------------------------------

    if method == "sbert":

        from lane3_retrieval_modern.sbert_faiss import (
            retrieve_with_scores,
        )

        candidates = retrieve_with_scores(claim, k)

    # --------------------------------------------------------
    # Modern retrieval: Cross-Encoder reranking
    # --------------------------------------------------------

    else:

        from lane3_retrieval_modern.reranker import search

        first_stage = (
            "sbert"
            if method == "sbert_rerank"
            else "bm25"
        )

        reranked = search(
            claim,
            top_n=k,
            first_stage=first_stage,
        )

        candidates = [
            (
                item["evidence_id"],
                item["rerank_score"],
                item["text"],
            )
            for item in reranked
        ]

    # Semantic similarity and Cross-Encoder scores can
    # have different scales. Do not apply the BM25 > 0 filter.

    return [
        (evidence_id, numeric_score, text)
        for evidence_id, score, text in candidates
        if math.isfinite(numeric_score := float(score))
    ]


def analyse_claim(claim: str, method: str, top_k: int):
    """Retrieve evidence and obtain evidence-level NLI predictions."""

    retrieved = retrieve_evidence(
        claim,
        top_k,
        method,
    )

    if not retrieved:
        return []

    from lane4_verification.nli_model import verify_batch

    predictions = verify_batch(
        claims=[claim] * len(retrieved),
        evidences=[
            text for _, _, text in retrieved
        ],
        batch_size=8,
    )

    if len(predictions) != len(retrieved):
        raise ValueError(
            "NLI returned a different number of "
            "predictions than evidence items"
        )

    results = []

    for (evidence_id, score, text), prediction in zip(
        retrieved,
        predictions,
    ):

        label = prediction["label"]

        if label not in LABELS:
            raise ValueError(
                f"Unexpected NLI label: {label}"
            )

        confidence = float(prediction["confidence"])

        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(
                "NLI confidence must be between 0 and 1"
            )

        results.append(
            {
                "evidence_id": evidence_id,
                "text": text,
                "retrieval_score": score,
                "label": label,
                "confidence": confidence,
            }
        )

    return results


def label_counts(results):
    """Count evidence-level NLI relationships."""

    counts = {
        label: 0
        for label in LABELS
    }

    for result in results:
        counts[result["label"]] += 1

    return counts


def explain_label(label: str) -> str:
    """Provide a user-friendly explanation for an NLI label."""

    return {
        "SUPPORTS": (
            "The NLI model considers this sentence "
            "supportive of the claim."
        ),
        "REFUTES": (
            "The NLI model considers this sentence "
            "contradictory to the claim."
        ),
        "NOT ENOUGH INFO": (
            "The NLI model does not find a clear supporting "
            "or contradicting relationship in this sentence."
        ),
    }[label]
