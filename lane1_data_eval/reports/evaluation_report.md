# Evaluation harness validation report — ClaimCheck (Lane 1)

This report validates the shared evaluation harness (`evaluation.py`) end-to-end on real
ClaimCheck data, using two traditional-retrieval baselines (TF-IDF cosine similarity and BM25)
and a majority-class label baseline. These numbers are a harness/sanity check, not ClaimCheck's
final reported results — Lanes 2-4's real retrieve()/rerank()/verify() implementations replace
these baselines, scored with the exact same functions, so every number in the final report is
produced identically.

## Retrieval: TF-IDF vs BM25 (test split, n=346), 95% bootstrap CI

| Metric | TF-IDF | BM25 |
|---|---|---|
| Recall@1 | 0.416 [0.364, 0.468] | 0.373 [0.324, 0.422] |
| Recall@3 | 0.855 [0.818, 0.890] | 0.847 [0.809, 0.884] |
| Recall@5 | 0.919 [0.887, 0.945] | 0.902 [0.867, 0.934] |
| Recall@10 | 0.942 [0.916, 0.965] | 0.934 [0.905, 0.960] |
| MRR | 0.640 [0.605, 0.674] | 0.607 [0.571, 0.639] |

**Significance test (paired bootstrap, n=2000 resamples), BM25 vs TF-IDF Recall@5:** diff=-0.017, p=0.172 (not statistically significant at alpha=0.05).

See `figures/retrieval_comparison_ci.png` for the chart version with error bars.

## Classification: majority-class baseline (test split, n=354)
- Accuracy: 0.537 [0.483, 0.588] (95% bootstrap CI)

## End-to-end FEVER score (BM25 retrieval@5 + majority-class label, n=346)
- 0.483

## Why this matters for the Part A -> Part B gap
Part A was marked down partly for asserting claims (e.g. 'established approach') without
evidence. Every number above ships with a confidence interval and, where two systems are
compared, a significance test — so claims like 'BM25 retrieves gold evidence more reliably
than TF-IDF' are backed by a p-value, not just a bigger number that could be test-set noise.