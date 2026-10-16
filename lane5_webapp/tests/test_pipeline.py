
"""Basic integration tests for ClaimCheck Lane 5."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane2_retrieval_traditional.retrieval import (
    retrieve_with_scores,
)
from lane4_verification.nli_model import verify_batch


CLAIM = "Telemundo is a Spanish-language television network."


@pytest.mark.parametrize("method", ["bm25", "tfidf"])
def test_retrieval_methods(method):
    """Both retrieval methods must return the expected structure."""
    results = retrieve_with_scores(CLAIM, k=5, method=method)

    assert isinstance(results, list)
    assert len(results) <= 5
    assert len(results) > 0

    for evidence_id, score, text in results:
        assert isinstance(evidence_id, str)
        assert isinstance(score, float)
        assert isinstance(text, str)
        assert text.strip()


def test_retrieval_contains_telemundo_evidence():
    """The known corpus should retrieve relevant Telemundo evidence."""
    results = retrieve_with_scores(CLAIM, k=5, method="bm25")
    ids = [item[0] for item in results]

    assert "ev_501" in ids
    assert "ev_500" in ids


def test_nli_returns_expected_structure():
    """The verifier must return one prediction per evidence item."""
    evidence = [
        "Telemundo is an American Spanish-language television network.",
        "Telemundo is an American English-language television network.",
    ]

    predictions = verify_batch(
        claims=[CLAIM] * len(evidence),
        evidences=evidence,
        batch_size=2,
    )

    assert len(predictions) == 2

    for prediction in predictions:
        assert prediction["label"] in {
            "SUPPORTS",
            "REFUTES",
            "NOT ENOUGH INFO",
        }
        assert 0.0 <= prediction["confidence"] <= 1.0


def test_nli_supports_and_refutes():
    """Known opposing evidence should yield opposing NLI labels."""
    evidence = [
        "Telemundo is an American Spanish-language television network.",
        "Telemundo is an American English-language television network.",
    ]

    predictions = verify_batch(
        claims=[CLAIM, CLAIM],
        evidences=evidence,
        batch_size=2,
    )

    assert predictions[0]["label"] == "SUPPORTS"
    assert predictions[1]["label"] == "REFUTES"
