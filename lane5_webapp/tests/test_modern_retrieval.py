
"""Fast tests for Lane 3 retrieval adapters."""

from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane5_webapp import pipeline


CLAIM = "Telemundo is a Spanish-language television network."


def test_modern_methods_available():

    assert {
        "sbert",
        "sbert_rerank",
        "bm25_rerank",
    } <= set(pipeline.METHOD_LABELS)


def test_sbert_keeps_negative_cosine_scores():

    with patch.object(
        pipeline,
        "load_retrieval_index",
    ), patch(
        "lane3_retrieval_modern.sbert_faiss.retrieve_with_scores",
        return_value=[
            ("ev_1", -0.1, "some evidence")
        ],
    ):

        results = pipeline.retrieve_evidence(
            CLAIM,
            5,
            "sbert",
        )

    assert results == [
        ("ev_1", -0.1, "some evidence")
    ]


@pytest.mark.parametrize(
    "method,first_stage",
    [
        ("sbert_rerank", "sbert"),
        ("bm25_rerank", "bm25"),
    ],
)
def test_reranker_adapter(method, first_stage):

    hits = [
        {
            "evidence_id": "ev_1",
            "rerank_score": 0.73,
            "text": "some evidence",
        }
    ]

    with patch.object(
        pipeline,
        "load_retrieval_index",
    ), patch(
        "lane3_retrieval_modern.reranker.search",
        return_value=hits,
    ) as search:

        results = pipeline.retrieve_evidence(
            CLAIM,
            5,
            method,
        )

    search.assert_called_once_with(
        CLAIM,
        top_n=5,
        first_stage=first_stage,
    )

    assert results == [
        ("ev_1", 0.73, "some evidence")
    ]


def test_semantic_result_reaches_nli(monkeypatch):

    hits = [
        (
            "ev_5",
            0.8,
            "a relevant evidence sentence",
        )
    ]

    monkeypatch.setattr(
        pipeline,
        "retrieve_evidence",
        lambda *args: hits,
    )

    import types

    fake = types.ModuleType(
        "lane4_verification.nli_model"
    )

    fake.verify_batch = lambda **kwargs: [
        {
            "label": "SUPPORTS",
            "confidence": 0.92,
        }
    ]

    monkeypatch.setitem(
        sys.modules,
        "lane4_verification.nli_model",
        fake,
    )

    result = pipeline.analyse_claim(
        CLAIM,
        "sbert",
        5,
    )

    assert result[0]["label"] == "SUPPORTS"
    assert result[0]["evidence_id"] == "ev_5"
