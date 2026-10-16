import pytest
from lane4_verification.claim_verdict import aggregate_predictions, verify_claim_from_evidence


def p(label, confidence=0.8):
    return {"label": label, "confidence": confidence}


def test_empty_abstains():
    r = aggregate_predictions([])
    assert (r["label"], r["confidence"]) == ("NOT ENOUGH INFO", 0)


def test_support_and_refute_separate():
    assert aggregate_predictions([p("SUPPORTS"), p("NOT ENOUGH INFO")])["label"] == "SUPPORTS"
    assert aggregate_predictions([p("REFUTES")])["label"] == "REFUTES"


def test_conflict_abstains():
    r = aggregate_predictions([p("SUPPORTS"), p("REFUTES")])
    assert r["label"] == "NOT ENOUGH INFO" and r["confidence"] == 0


def test_top1_uses_first():
    r = aggregate_predictions([p("REFUTES", .7), p("SUPPORTS", .9)], strategy="top1")
    assert r["label"] == "REFUTES" and r["confidence"] == .7


@pytest.mark.parametrize("wrong", ["UNKNOWN", "", "supports"])
def test_bad_label_rejected(wrong):
    with pytest.raises(ValueError):
        aggregate_predictions([p(wrong)])


def test_bad_confidence_rejected():
    with pytest.raises(ValueError):
        aggregate_predictions([p("SUPPORTS", 1.5)])


def test_bad_strategy_rejected_even_when_empty():
    with pytest.raises(ValueError):
        aggregate_predictions([], strategy="made_up")


def test_empty_claim_rejected():
    with pytest.raises(ValueError):
        verify_claim_from_evidence(" ", [])


def test_empty_evidence_skips_model():
    assert verify_claim_from_evidence("Claim", [])["verdict"]["label"] == "NOT ENOUGH INFO"


def test_integration_mock(monkeypatch):
    import sys
    from types import ModuleType
    stub = ModuleType("lane4_verification.nli_model")
    stub.verify_batch = lambda claims, evidences, batch_size=8: [p("SUPPORTS") for _ in evidences]
    monkeypatch.setitem(sys.modules, "lane4_verification.nli_model", stub)
    out = verify_claim_from_evidence("A claim", [{"evidence_id": "ev_1", "text": "A fact"}])
    assert out["verdict"]["label"] == "SUPPORTS"
    assert out["evidence"][0]["evidence_id"] == "ev_1"
