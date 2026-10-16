"""Isolated interface tests: no large model download required."""
import sys
from types import ModuleType
import pytest
from lane4_verification import retrieval_verification as rv

CLAIM = 'Telemundo is a Spanish-language television network.'


def stub(monkeypatch, fullname, **funcs):
    mod = ModuleType(fullname)
    for name, func in funcs.items():
        setattr(mod, name, func)
    monkeypatch.setitem(sys.modules, fullname, mod)
    return mod


@pytest.mark.parametrize('method', ['bm25', 'tfidf'])
def test_lane2_methods(monkeypatch, method):
    calls = []
    stub(monkeypatch, 'lane2_retrieval_traditional.retrieval',
         retrieve_with_scores=lambda claim, k, method: calls.append((k, method)) or
         [('ev_1', 2.3, 'alpha'), ('ev_2', 0, 'irrelevant')])
    out = rv.retrieve_ranked_evidence(CLAIM, method=method, k=5)
    assert calls == [(5, method)]
    assert [x['evidence_id'] for x in out] == ['ev_1']
    assert out[0]['retrieval_score'] == 2.3


def test_sbert_preserves_negative_similarity(monkeypatch):
    stub(monkeypatch, 'lane3_retrieval_modern.sbert_faiss',
         retrieve_with_scores=lambda claim, k: [('ev_3', -.11, 'evidence')])
    out = rv.retrieve_ranked_evidence(CLAIM, method='sbert')
    assert len(out) == 1 and out[0]['retrieval_score'] == -.11


@pytest.mark.parametrize('method,stage', [('sbert_rerank', 'sbert'), ('bm25_rerank', 'bm25')])
def test_reranker_stages(monkeypatch, method, stage):
    calls = []
    stub(monkeypatch, 'lane3_retrieval_modern.reranker',
         search=lambda claim, top_n, first_stage: calls.append((top_n, first_stage)) or [
             dict(evidence_id='ev_4', text='evidence', rerank_score=.71, retrieval_score=.13)])
    out = rv.retrieve_ranked_evidence(CLAIM, method=method, k=3)
    assert calls == [(3, stage)]
    assert out[0]['first_stage_score'] == .13
    assert out[0]['retrieval_score'] == .71


def test_end_to_end_without_model_download(monkeypatch):
    monkeypatch.setattr(rv, 'retrieve_ranked_evidence', lambda *a, **kw: [
        dict(evidence_id='ev_1', text='positive', retrieval_score=.8),
        dict(evidence_id='ev_2', text='negative', retrieval_score=.7)])
    stub(monkeypatch, 'lane4_verification.nli_model',
         verify_batch=lambda claims, evidences, batch_size: [
             {'label': 'SUPPORTS', 'confidence': .91},
             {'label': 'REFUTES', 'confidence': .88}])
    result = rv.verify_retrieved_claim(CLAIM, method='sbert_rerank', k=2)
    assert [e['evidence_id'] for e in result['evidence']] == ['ev_1', 'ev_2']
    assert result['verdict']['label'] == 'NOT ENOUGH INFO'
    assert result['verdict']['rationale'].startswith('Conflicting')
    assert result['method'] == 'sbert_rerank'


@pytest.mark.parametrize('method,k,claim', [('other', 5, CLAIM), ('bm25', 0, CLAIM), ('bm25', 5, '')])
def test_rejects_invalid_inputs(method, k, claim):
    with pytest.raises(ValueError):
        rv.retrieve_ranked_evidence(claim, method=method, k=k)


def test_empty_returns_abstention(monkeypatch):
    monkeypatch.setattr(rv, 'retrieve_ranked_evidence', lambda *a, **kw: [])
    out = rv.verify_retrieved_claim(CLAIM, method='bm25')
    assert out['evidence'] == []
    assert out['verdict']['confidence'] == 0


def test_deduplicates_and_discards_invalid(monkeypatch):
    stub(monkeypatch, 'lane2_retrieval_traditional.retrieval',
         retrieve_with_scores=lambda claim, k, method: [
             ('ev_1', 2.0, 'text'), ('ev_1', 1.0, 'duplicate'),
             ('ev_2', float('nan'), 'bad'), ('ev_3', 1., '')])
    assert len(rv.retrieve_ranked_evidence(CLAIM, method='bm25')) == 1
