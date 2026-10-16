
"""Tests for Lane 5's use of Lane 4 claim aggregation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane5_webapp.pipeline import claim_level_prediction


def test_top1_keeps_first_prediction():
    predictions = [
        {
            "evidence_id": "ev_501",
            "label": "SUPPORTS",
            "confidence": 0.91,
        },
        {
            "evidence_id": "ev_500",
            "label": "REFUTES",
            "confidence": 0.99,
        },
    ]

    verdict = claim_level_prediction(predictions)

    assert verdict["label"] == "SUPPORTS"
    assert verdict["confidence"] == 0.91
    assert verdict["strategy"] == "top1"
    assert verdict["counts"]["REFUTES"] == 1


def test_empty_evidence_abstains():
    verdict = claim_level_prediction([])

    assert verdict["label"] == "NOT ENOUGH INFO"
    assert verdict["confidence"] == 0


def test_neutral_top_rank_is_nei():
    predictions = [
        {
            "evidence_id": "ev_1",
            "label": "NOT ENOUGH INFO",
            "confidence": 0.7,
        },
        {
            "evidence_id": "ev_2",
            "label": "SUPPORTS",
            "confidence": 0.9,
        },
    ]

    verdict = claim_level_prediction(predictions)

    assert verdict["label"] == "NOT ENOUGH INFO"
