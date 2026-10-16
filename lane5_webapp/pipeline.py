"""Reusable retrieval + evidence-level verification for ClaimCheck Lane 5.

No Streamlit UI is imported or executed here. Dependencies are injected in tests.
The eventual claim-level verdict must come from an evaluated team decision rule.
"""
from __future__ import annotations

import math
from functools import lru_cache

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
METHOD_LABELS = {
    "bm25": "BM25 — lexical retrieval",
    "tfidf": "TF-IDF — vector-space retrieval",
}


@lru_cache(maxsize=2)
def load_retrieval_index(method: str):
    """Build/reuse the Lane 2 index once per method per Python process."""
    if method not in METHOD_LABELS:
        raise ValueError(f"Unsupported retrieval method: {method}")
    from lane2_retrieval_traditional.retrieval import get_index
    return get_index(method)


def retrieve_evidence(claim: str, k: int, method: str):
    """Get (evidence_id, score, text) tuples, excluding non-positive scores.

    Positive-score filtering is a *heuristic*, not a tested relevance threshold.
    """
    if method not in METHOD_LABELS:
        raise ValueError(f"Unsupported retrieval method: {method}")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Claim cannot be empty")

    from lane2_retrieval_traditional.retrieval import retrieve_with_scores
    load_retrieval_index(method)
    candidates = retrieve_with_scores(claim=claim, k=k, method=method)
    return [
        (evidence_id, numeric_score, text)
        for evidence_id, score, text in candidates
        if math.isfinite(numeric_score := float(score)) and numeric_score > 0
    ]


def analyse_claim(claim: str, method: str, top_k: int):
    """Return ranked evidence records with per-sentence NLI labels."""
    retrieved = retrieve_evidence(claim, top_k, method)
    if not retrieved:
        return []

    from lane4_verification.nli_model import verify_batch
    predictions = verify_batch(
        claims=[claim] * len(retrieved),
        evidences=[text for _, _, text in retrieved],
        batch_size=8,
    )
    if len(predictions) != len(retrieved):
        raise ValueError("NLI returned a different number of predictions than evidence items")

    results = []
    for (evidence_id, score, text), prediction in zip(retrieved, predictions):
        label = prediction["label"]
        if label not in LABELS:
            raise ValueError(f"Unexpected NLI label: {label}")
        confidence = float(prediction["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("NLI confidence must be between 0 and 1")
        results.append({
            "evidence_id": evidence_id,
            "text": text,
            "retrieval_score": score,
            "label": label,
            "confidence": confidence,
        })
    return results


def label_counts(results):
    counts = {label: 0 for label in LABELS}
    for result in results:
        counts[result["label"]] += 1
    return counts


def explain_label(label: str) -> str:
    return {
        "SUPPORTS": "The NLI model considers this sentence supportive of the claim.",
        "REFUTES": "The NLI model considers this sentence contradictory to the claim.",
        "NOT ENOUGH INFO": (
            "The NLI model does not find a clear supporting or "
            "contradicting relationship in this sentence."
        ),
    }[label]
