"""Experimental claim-level verdicts from individual NLI evidence predictions.

No training or hidden thresholds. NOT an officially validated FEVER verdict.
Lane 5 can import classify_evidence / aggregate_predictions independently.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
NEI = "NOT ENOUGH INFO"


def aggregate_predictions(predictions: Sequence[dict], *, strategy: str = "conflict_abstain") -> dict:
    """Return claim-level {label, confidence, rationale, counts, strategy}.

    conflict_abstain: abstain (NEI) if both SUPPORTS and REFUTES evidence.
    top1: use first retrieved evidence prediction (for comparison only).
    confidence on NEI due to conflict/no evidence is 0, not model certainty.
    Positive decision confidence is the max NLI softmax of that label; not
    calibrated factual correctness.
    """
    if strategy not in {"conflict_abstain", "top1"}:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    counts = {label: 0 for label in ("SUPPORTS", "REFUTES", NEI)}
    for pred in predictions:
        label = pred["label"]
        if label not in LABELS:
            raise ValueError(f"Unknown NLI label: {label}")
        confidence = float(pred["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("NLI confidence must be within [0, 1]")
        counts[label] += 1

    if not predictions:
        return dict(label=NEI, confidence=0.0, rationale="No retrieved evidence.", counts=counts, strategy=strategy)

    if strategy == "top1":
        first = predictions[0]
        return dict(label=first["label"], confidence=float(first["confidence"]),
                    rationale="Top-ranked evidence only; not an evaluated final verdict.",
                    counts=counts, strategy=strategy)

    if counts["SUPPORTS"] and counts["REFUTES"]:
        label, confidence, rationale = NEI, 0.0, "Conflicting supporting and refuting evidence; abstained."
    elif counts["SUPPORTS"]:
        label = "SUPPORTS"
        confidence = max(float(x["confidence"]) for x in predictions if x["label"] == label)
        rationale = "At least one evidence sentence supports, none refutes."
    elif counts["REFUTES"]:
        label = "REFUTES"
        confidence = max(float(x["confidence"]) for x in predictions if x["label"] == label)
        rationale = "At least one evidence sentence refutes, none supports."
    else:
        label, confidence, rationale = NEI, 0.0, "No supporting or refuting evidence identified."

    return dict(label=label, confidence=confidence, rationale=rationale,
                counts=counts, strategy=strategy)


def verify_claim_from_evidence(claim: str, evidence: Sequence[dict], *,
                               strategy: str = "conflict_abstain", batch_size: int = 8) -> dict:
    """Verify caller-supplied ranked evidence; does NOT retrieve evidence.

    evidence items: [{evidence_id, text, retrieval_score?}, ...].
    Returns one output containing evidence-level predictions + aggregate.
    """
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    if not evidence:
        return {"claim": claim, "evidence": [], "verdict": aggregate_predictions([], strategy=strategy)}
    from .nli_model import verify_batch
    preds = verify_batch([claim] * len(evidence), [e["text"] for e in evidence], batch_size=batch_size)
    if len(preds) != len(evidence):
        raise ValueError("Evidence and NLI prediction counts differ")
    enriched = [{**dict(item), **dict(pred)} for item, pred in zip(evidence, preds)]
    return {"claim": claim, "evidence": enriched,
            "verdict": aggregate_predictions(preds, strategy=strategy)}
