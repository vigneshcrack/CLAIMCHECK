'''
Zero-shot NLI verifier with label-mapping layer

Checkpoint: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

Interface contract:
    verify(claim: str, evidence: str) -> {"label": ..., "confidence": float}

'''

import re
import sys
from pathlib import Path
 
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
 
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
# NLI label -> Project label
NLI_TO_FEVER = {
    "entailment": "SUPPORTS",
    "contradiction": "REFUTES",
    "neutral": "NOT ENOUGH INFO",
}
 
NEI = "NOT ENOUGH INFO"
EVIDENCE_SEP = " "  # Original top-k concatenation convention

# Text cleaning: removing bracket escape codes left over from FEVER dataset tokenization.

_BRACKETS = {
    "-LRB-": "(", "-RRB-": ")",
    "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}",
}

def clean_fever_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    for k, v in _BRACKETS.items():
        t = t.replace(k, v)
    t = re.sub(r"\s+([.,;:!?%)\]])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    t = re.sub(r"\s+('s|n't)\b", r"\1", t)
    t = t.replace("``", '"').replace("''", '"')
    return re.sub(r"\s+", " ", t).strip()

# Model loading

_tok = None
_model = None
_idx2fever = None
 
 
def _load():
    global _tok, _model, _idx2fever
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = (
            AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            .to(DEVICE)
            .eval()
        )
        _idx2fever = {
            i: NLI_TO_FEVER[label.lower()]
            for i, label in _model.config.id2label.items()
        }

# Inference

@torch.no_grad()
def verify_batch(claims, evidences, batch_size=32):
    if len(claims) != len(evidences):
        raise ValueError("claims and evidences must have the same length")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    claims = [clean_fever_text(c) for c in claims]
    evidences = [clean_fever_text(e) for e in evidences]
 
    results = [None] * len(claims)
 
    todo = []
    for i, e in enumerate(evidences):
        if not e:
            results[i] = {"label": NEI, "confidence": 0.0}  # empty-input fallback, not a model probability
        else:
            todo.append(i)
 
    if todo:
        _load()

    for start in range(0, len(todo), batch_size):
        idxs = todo[start:start + batch_size]
        enc = _tok(
            [evidences[i] for i in idxs],
            [claims[i] for i in idxs],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        probs = torch.softmax(_model(**enc).logits, dim=-1).cpu()
        conf, pred = probs.max(dim=-1)
        for i, k, c in zip(idxs, pred, conf):
            results[i] = {"label": _idx2fever[int(k)], "confidence": float(c)}
 
    return results

def verify(claim: str, evidence: str) -> dict:
    """Return {"label": "SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO", "confidence": float}."""
    return verify_batch([claim], [evidence])[0]

# Evidence retrieval and combination into one string for verification

def _import_retrieval(method="bm25"):
    """Select Lane 2 lexical retrieval explicitly, not a path-dependent module."""
    if method not in ("bm25", "tfidf"):
        raise ValueError("This legacy wrapper supports bm25 or tfidf only")
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lane2_retrieval_traditional.retrieval import retrieve_with_scores
    return retrieve_with_scores


def retrieve_evidence(claim: str, k: int = 5, sep: str = EVIDENCE_SEP, method="bm25") -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    retrieve_with_scores = _import_retrieval(method)
    hits = retrieve_with_scores(claim, k=k, method=method)
    sentences = [clean_fever_text(h[2]) for h in hits]
    return {
        "evidence": sep.join(s for s in sentences if s),
        "evidence_ids": [h[0] for h in hits],
    }


def verify_with_retrieval(claim: str, k: int = 5,
                          sep: str = EVIDENCE_SEP, method="bm25") -> dict:
    """EXPERIMENTAL: verify one concatenated top-k evidence string."""
    ev = retrieve_evidence(claim, k, sep, method)
    return {**verify(claim, ev["evidence"]), **ev}


def verify_with_retrieval_batch(claims, k: int = 5,
                                sep: str = EVIDENCE_SEP, method="bm25") -> list:
    """EXPERIMENTAL: concatenated top-k NLI on several claims."""
    claims = list(claims)
    evs = [retrieve_evidence(c, k, sep, method) for c in claims]
    preds = verify_batch(claims, [e["evidence"] for e in evs])
    return [{**p, **e} for p, e in zip(preds, evs)]

if __name__ == "__main__":
    _load()
    print("id2label:", _model.config.id2label)
    print("mapping :", _idx2fever)
    tests = [
        ("Paris is the capital of France.",
         "Paris is the capital and largest city of France."),            # SUPPORTS
        ("Paris is the capital of Germany.",
         "Paris is the capital and largest city of France."),            # REFUTES
        ("Paris is the capital of France.",
         "Bananas are rich in potassium."),                              # NEI
        ("Telemundo is a Spanish-language network.",
         "Telemundo is an American English-language television network."),  # REFUTES
        ("Telemundo is a Spanish-language network.",
         "Telemundo is an American Spanish-language television network."),  # SUPPORTS
    ]
    for c, e in tests:
        print(verify(c, e), "|", c)

