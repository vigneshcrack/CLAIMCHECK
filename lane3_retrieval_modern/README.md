# Lane 3 — Modern Retrieval (Sentence-BERT + FAISS + reranking)

Owner: _(assign teammate name here)_

## What you're building

A `retrieve(claim, k)` function using semantic embeddings (Sentence-BERT + FAISS), plus a `rerank(claim, candidates, top_n)` function using a cross-encoder — scored against the shared evaluation harness in `../lane1_data_eval/scripts/evaluation.py`.

## Shared interface contract — implement exactly these signatures

```python
def retrieve(claim: str, k: int) -> list[str]:
    """Return the top-k evidence_id strings, ranked best-first."""

def rerank(claim: str, candidates: list[str], top_n: int) -> list[str]:
    """Re-order/filter candidate evidence_ids, best-first, down to top_n."""
```

## Data you need (already built by Lane 1 — don't rebuild it)

- `../lane1_data_eval/data/processed/claimcheck_dataset.csv`
- `../lane1_data_eval/data/processed/evidence_corpus.csv`

## Steps

1. Pick ONE Sentence-BERT checkpoint (e.g. `all-mpnet-base-v2`); state why (quality/speed tradeoff) in 2-3 sentences.
2. Embed `evidence_corpus.csv["text"]`; build a FAISS index.
3. Implement `retrieve(claim, k)` matching the signature above.
4. Add reranking: pick ONE cross-encoder checkpoint (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`); implement `rerank(claim, candidates, top_n)`.
5. Score yourself with Lane 1's harness (same import pattern as Lane 2's README) — directly compare your Recall@k / MRR against Lane 2's traditional baseline using `bootstrap_ci` and `paired_bootstrap_test` (already built — this answers RQ1: does semantic retrieval beat lexical retrieval, with a significance test, not just a bigger number).
6. Write a methods paragraph: checkpoints named, embedding dimension, top-K before rerank, top-N after.

## Submit to the group

- `retrieval.py` with `retrieve(claim, k)`
- `reranker.py` with `rerank(claim, candidates, top_n)`
- FAISS index build script
- Methods paragraph with named checkpoints + justification
- Output for the same 5 sample claims as Lane 2

## Commit discipline

Push to this folder daily, even small progress — commit history is evidence of individual contribution for SPARKPlus and the report's "contributions by team member" section.
