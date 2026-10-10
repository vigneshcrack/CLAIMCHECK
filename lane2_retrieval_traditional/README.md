# Lane 2 — Traditional Retrieval (BM25 / TF-IDF)

Owner: _(assign teammate name here)_

## What you're building

A `retrieve(claim, k)` function using a classic lexical retrieval method (BM25 or TF-IDF), scored against the shared evaluation harness in `../lane1_data_eval/scripts/evaluation.py`.

## Shared interface contract — implement exactly this signature

```python
def retrieve(claim: str, k: int) -> list[str]:
    """Return the top-k evidence_id strings (e.g. 'ev_42'), ranked best-first."""
```

## Data you need (already built by Lane 1 — don't rebuild it)

- `../lane1_data_eval/data/processed/claimcheck_dataset.csv` — claims + gold labels + `gold_evidence_id`
- `../lane1_data_eval/data/processed/evidence_corpus.csv` — the `evidence_id, text` pairs you retrieve over

## Steps

1. Build a BM25 or TF-IDF index over `evidence_corpus.csv["text"]` (there's a working BM25 + TF-IDF example already in `../lane1_data_eval/scripts/evaluation.py` — reuse that pattern, don't start from zero).
2. Implement `retrieve(claim, k)` matching the signature above.
3. Sanity-check on 5 sample claims manually before handing off.
4. Score yourself with Lane 1's harness:
   ```python
   import sys; sys.path.insert(0, "../lane1_data_eval")
   from scripts.evaluation import recall_at_k, mean_reciprocal_rank, bootstrap_ci
   ```
5. Write a short methods paragraph: library used, parameters (e.g. BM25's k1/b), why this counts as the "traditional" baseline.

## Submit to the group

- `retrieval.py` with `retrieve(claim, k)`
- Index build script
- Methods paragraph (parameters + rationale) for the report
- Output for the same 5 sample claims Lane 3 will also run, for a side-by-side sanity check

## Commit discipline

Push to this folder daily, even small progress — commit history is evidence of individual contribution for SPARKPlus and the report's "contributions by team member" section.
