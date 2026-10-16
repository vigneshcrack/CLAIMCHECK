
"""Fast Lane 5 pipeline unit tests."""

from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane5_webapp import pipeline


CLAIM = "Telemundo is a Spanish-language television network."


def test_invalid_retrieval_method():

    with pytest.raises(
        ValueError,
        match="Unsupported retrieval method",
    ):
        pipeline.retrieve_evidence(
            CLAIM,
            5,
            "unknown",
        )


@pytest.mark.parametrize(
    "claim,k",
    [
        ("", 5),
        ("   ", 5),
        (CLAIM, 0),
        (CLAIM, -2),
    ],
)
def test_invalid_inputs(claim, k):

    with pytest.raises(ValueError):

        pipeline.retrieve_evidence(
            claim,
            k,
            "bm25",
        )


def test_zero_and_negative_scores_are_filtered():

    candidates = [
        ("ev_a", 4.2, "Relevant"),
        ("ev_b", 0.0, "Zero"),
        ("ev_c", -2.0, "Negative"),
    ]

    with patch.object(
        pipeline,
        "load_retrieval_index",
    ), patch(
        "lane2_retrieval_traditional.retrieval.retrieve_with_scores",
        return_value=candidates,
    ):

        results = pipeline.retrieve_evidence(
            CLAIM,
            5,
            "bm25",
        )

    assert results == [
        ("ev_a", 4.2, "Relevant"),
    ]


def test_empty_retrieval_skips_nli():

    with patch.object(
        pipeline,
        "retrieve_evidence",
        return_value=[],
    ):

        assert pipeline.analyse_claim(
            CLAIM,
            "bm25",
            5,
        ) == []


def test_combined_predictions_and_conflict_counts():

    evidence = [
        ("ev_501", 8.0, "Spanish-language network"),
        ("ev_500", 7.0, "English-language network"),
    ]

    predictions = [
        {
            "label": "SUPPORTS",
            "confidence": 0.98,
        },
        {
            "label": "REFUTES",
            "confidence": 0.96,
        },
    ]

    with patch.object(
        pipeline,
        "retrieve_evidence",
        return_value=evidence,
    ), patch(
        "lane4_verification.nli_model.verify_batch",
        return_value=predictions,
    ):

        results = pipeline.analyse_claim(
            CLAIM,
            "bm25",
            5,
        )

    assert [
        x["evidence_id"]
        for x in results
    ] == [
        "ev_501",
        "ev_500",
    ]

    assert pipeline.label_counts(results) == {
        "SUPPORTS": 1,
        "REFUTES": 1,
        "NOT ENOUGH INFO": 0,
    }


def test_mismatched_prediction_count_raises():

    with patch.object(
        pipeline,
        "retrieve_evidence",
        return_value=[
            ("ev_1", 1.0, "text")
        ],
    ), patch(
        "lane4_verification.nli_model.verify_batch",
        return_value=[],
    ):

        with pytest.raises(
            ValueError,
            match="different number",
        ):

            pipeline.analyse_claim(
                CLAIM,
                "bm25",
                5,
            )


def test_invalid_model_label_raises():

    with patch.object(
        pipeline,
        "retrieve_evidence",
        return_value=[
            ("ev_1", 1.0, "text")
        ],
    ), patch(
        "lane4_verification.nli_model.verify_batch",
        return_value=[
            {
                "label": "UNKNOWN",
                "confidence": 0.5,
            }
        ],
    ):

        with pytest.raises(
            ValueError,
            match="Unexpected NLI label",
        ):

            pipeline.analyse_claim(
                CLAIM,
                "bm25",
                5,
            )


def test_model_failure_propagates():

    with patch.object(
        pipeline,
        "retrieve_evidence",
        return_value=[
            ("ev_1", 1.0, "text")
        ],
    ), patch(
        "lane4_verification.nli_model.verify_batch",
        side_effect=RuntimeError(
            "Model unavailable"
        ),
    ):

        with pytest.raises(
            RuntimeError,
            match="Model unavailable",
        ):

            pipeline.analyse_claim(
                CLAIM,
                "bm25",
                5,
            )
