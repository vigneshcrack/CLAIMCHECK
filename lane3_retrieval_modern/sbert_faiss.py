"""
sbert_faiss.py — Lane 3: modern semantic retrieval (Sentence-BERT + FAISS) for ClaimCheck.

Shared interface contract:

    def retrieve(claim: str, k: int) -> list[str]
        Return the top-k evidence_id strings (e.g. 'ev_42'), ranked best-first.

Extras (same shapes as Lane 2, so the app can swap retrievers with one line):

    retrieve_with_scores(claim, k) -> [(evidence_id, cosine_score, readable_text), ...]
    retrieve_batch(claims, k)      -> [[evidence_id, ...], ...]   (fast path for evaluation)
    get_text(evidence_id)          -> readable evidence sentence

Data (built by Lane 1, not rebuilt here):
    ../lane1_data_eval/data/processed/evidence_corpus.csv   (evidence_id, text)

Design:
  * Checkpoint: sentence-transformers/all-mpnet-base-v2 (768-d, mean pooling).
  * Evidence text is cleaned of FEVER escape tokens (-LRB- -> "(") before embedding,
    because a transformer sees "-LRB-" as noise, unlike BM25's tokenizer which strips it.
  * Embeddings are L2-normalised and stored in a FAISS IndexFlatIP, so inner product = cosine
    similarity. Flat (exact) search is used because the corpus is small (648 sentences):
    approximate indexes (IVF/HNSW) only pay off at ~100k+ vectors and would add recall loss.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CORPUS_CSV = Path(os.environ.get(
    "CLAIMCHECK_CORPUS_CSV",
    HERE.parent / "lane1_data_eval" / "data" / "processed" / "evidence_corpus.csv"))
INDEX_DIR = HERE / "indexes"

SBERT_MODEL = os.environ.get("CLAIMCHECK_SBERT_MODEL", "sentence-transformers/all-mpnet-base-v2")
BATCH_SIZE = 64

# ----------------------------------------------------------------------------
# Text cleaning (same rules as Lane 2's clean_text, kept local so Lane 3 has no Lane 2 dependency)
# ----------------------------------------------------------------------------
_FEVER_ESCAPES = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
                  "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}


def clean_text(text: str) -> str:
    """Readable FEVER sentence: '-LRB-' -> '(', fix spacing around punctuation."""
    text = str(text).replace("``", '"').replace("''", '"')
    for a, b in _FEVER_ESCAPES.items():
        text = text.replace(a, b)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+('s|n't)\b", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------------
def load_corpus(path: str | Path = CORPUS_CSV) -> pd.DataFrame:
    """Lane 1's corpus, deduplicated exactly like Lane 2 so both lanes index the same evidence_ids."""
    df = pd.read_csv(path)
    if not {"evidence_id", "text"} <= set(df.columns):
        raise ValueError(f"{path} must have columns evidence_id, text — found {list(df.columns)}")
    df = df.dropna(subset=["evidence_id", "text"]).drop_duplicates("evidence_id").reset_index(drop=True)
    df["evidence_id"] = df["evidence_id"].astype(str)
    df["text"] = df["text"].astype(str)
    return df


def _fingerprint(df: pd.DataFrame, model_name: str) -> str:
    h = hashlib.sha1(model_name.encode())
    for eid, t in zip(df["evidence_id"], df["text"]):
        h.update(f"{eid}\t{t}\n".encode("utf-8"))
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
_MODELS: dict = {}


def get_encoder(model_name: str = SBERT_MODEL):
    if model_name not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS[model_name] = SentenceTransformer(model_name)
    return _MODELS[model_name]


def encode(texts: list[str], model_name: str = SBERT_MODEL) -> np.ndarray:
    emb = get_encoder(model_name).encode(list(texts), batch_size=BATCH_SIZE, convert_to_numpy=True,
                                         normalize_embeddings=True, show_progress_bar=False)
    return np.ascontiguousarray(emb, dtype="float32")


# ----------------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------------
class SbertFaissIndex:
    def __init__(self, model_name: str = SBERT_MODEL):
        self.model_name = model_name

    def fit(self, df: pd.DataFrame):
        import faiss
        self.ids = df["evidence_id"].tolist()
        self.texts = [clean_text(t) for t in df["text"]]
        emb = encode(self.texts, self.model_name)
        self.dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)
        self.fingerprint = _fingerprint(df, self.model_name)
        return self

    def save(self, out_dir: Path | None = None):
        import faiss
        out_dir = Path(out_dir or INDEX_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "sbert_faiss.index"))
        (out_dir / "sbert_meta.json").write_text(json.dumps(dict(
            model_name=self.model_name, dim=self.dim, n=len(self.ids), fingerprint=self.fingerprint,
            ids=self.ids, texts=self.texts), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path | None = None):
        import faiss
        out_dir = Path(out_dir or INDEX_DIR)
        meta = json.loads((out_dir / "sbert_meta.json").read_text(encoding="utf-8"))
        obj = cls(meta["model_name"])
        obj.index = faiss.read_index(str(out_dir / "sbert_faiss.index"))
        obj.ids, obj.texts, obj.dim, obj.fingerprint = meta["ids"], meta["texts"], meta["dim"], meta["fingerprint"]
        return obj

    def search(self, claims: list[str], k: int):
        k = max(1, min(int(k), len(self.ids)))
        q = encode([clean_text(c) for c in claims], self.model_name)
        scores, idx = self.index.search(q, k)
        return scores, idx


def build_index(corpus_csv: str | Path = CORPUS_CSV, model_name: str = SBERT_MODEL, save: bool = True):
    idx = SbertFaissIndex(model_name).fit(load_corpus(corpus_csv))
    if save:
        idx.save()
    return idx


_CACHE: dict = {}


def get_index(model_name: str = SBERT_MODEL) -> SbertFaissIndex:
    """Load the saved index; rebuild automatically if missing or if the corpus/model changed."""
    if model_name not in _CACHE:
        idx = None
        if (INDEX_DIR / "sbert_meta.json").exists() and (INDEX_DIR / "sbert_faiss.index").exists():
            try:
                cand = SbertFaissIndex.load()
                if cand.fingerprint == _fingerprint(load_corpus(), model_name):
                    idx = cand
            except Exception:
                idx = None
        if idx is None:
            idx = build_index(model_name=model_name, save=(model_name == SBERT_MODEL))
        _CACHE[model_name] = idx
    return _CACHE[model_name]


def get_text(evidence_id: str) -> str:
    idx = get_index()
    if not hasattr(idx, "_pos"):
        idx._pos = {e: i for i, e in enumerate(idx.ids)}
    return idx.texts[idx._pos[evidence_id]]


# ----------------------------------------------------------------------------
# Shared interface
# ----------------------------------------------------------------------------
def retrieve_batch(claims: list[str], k: int, model_name: str = SBERT_MODEL) -> list[list[str]]:
    idx = get_index(model_name)
    if not claims:
        return []
    _, I = idx.search(list(claims), k)
    return [[idx.ids[i] for i in row if i >= 0] for row in I]


def retrieve_with_scores(claim: str, k: int, model_name: str = SBERT_MODEL) -> list[tuple[str, float, str]]:
    """[(evidence_id, cosine_similarity, readable_text), ...] best-first. Cosine is in [-1, 1]."""
    idx = get_index(model_name)
    S, I = idx.search([claim], k)
    return [(idx.ids[i], float(s), idx.texts[i]) for s, i in zip(S[0], I[0]) if i >= 0]


def retrieve(claim: str, k: int) -> list[str]:
    """Return the top-k evidence_id strings (e.g. 'ev_42'), ranked best-first."""
    return [eid for eid, _, _ in retrieve_with_scores(claim, k)]
