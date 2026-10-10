# ClaimCheck

Evidence retrieval + claim verification system — 36118 Applied Natural Language Processing (ANLP), Spring 2026, AT2 Part B group project.

**Team:** Vignesh · Nandeeswari · Pranav · Sivadarshan · Delisiya

## What this is

ClaimCheck takes a factual claim and returns a verdict (SUPPORTS / REFUTES / NOT ENOUGH INFO) backed by retrieved evidence, comparing a traditional lexical retrieval pipeline against a modern semantic one. Built on FEVER-derived data (see `lane1_data_eval/README.md` for the dataset provenance note — important, read it before writing the report's Datasets section).

## Project structure — one folder per person, so everyone can work in parallel without blocking each other

| Folder | Lane | Owner |
|---|---|---|
| `lane1_data_eval/` | Data, Evaluation & Innovation Lead | Vignesh |
| `lane2_retrieval_traditional/` | Traditional Retrieval (BM25/TF-IDF) | _assign_ |
| `lane3_retrieval_modern/` | Modern Retrieval (SBERT + FAISS + rerank) | _assign_ |
| `lane4_verification/` | Verification / NLI | _assign_ |
| `lane5_webapp/` | Web App & Explainability | _assign_ |

Each lane folder has its own `README.md` with exact steps and deliverables. Start there.

## The shared interface contract (read this first — it's what lets 5 people build in parallel)

```python
def retrieve(claim: str, k: int) -> list[str]:
    """Return top-k evidence_id strings, ranked best-first."""

def rerank(claim: str, candidates: list[str], top_n: int) -> list[str]:
    """Re-order/filter candidate evidence_ids down to top_n, best-first."""

def verify(claim: str, evidence: str) -> dict:
    """Return {"label": "SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO", "confidence": float}."""
```

Lane 1 built the data (`lane1_data_eval/data/processed/`) and the scoring functions everyone uses to evaluate their own work (`lane1_data_eval/scripts/evaluation.py` — `recall_at_k`, `mean_reciprocal_rank`, `accuracy`, `macro_f1`, `confusion_matrix_df`, `fever_score`, plus `bootstrap_ci` and `paired_bootstrap_test` for confidence intervals and significance testing). Nobody else needs to re-implement metrics — import from there.

## Getting started (for teammates who haven't used git before)

**Easiest option: GitHub Desktop** (no command line needed) — download from desktop.github.com, sign in, "Clone a repository" using the URL Vignesh shares, and it gives you buttons for commit/push/pull.

**Command line option:**
```
git clone <repo-url>
cd claimcheck
```

Then work inside **your own lane folder only** — this avoids merge conflicts since nobody else is touching your files. When you've made progress:
```
git add <your-lane-folder>
git commit -m "describe what you did"
git push
```
Before starting work each day, pull the latest:
```
git pull
```

**Commit daily, even small progress.** Commit history is used as evidence of individual contribution, both for the report's "project phases and contributions by individual team members" section and for SPARKPlus peer evaluation.

## Running Lane 1's pipeline

```
cd lane1_data_eval
pip install -r requirements.txt
python -I run_pipeline.py
python -m pytest tests/ -v
```

See `lane1_data_eval/README.md` for full details.
