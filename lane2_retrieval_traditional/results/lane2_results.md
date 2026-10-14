# Lane 2 — Traditional retrieval results (scored with Lane 1's evaluation harness)

Corpus: 648 evidence sentences (`evidence_corpus.csv`). Test split: 346 claims with a gold evidence id (NEI claims excluded, as the harness requires).

## Test split: Recall@k and MRR, 95% bootstrap CI (n=346)

|           | Lane 2 BM25          | Lane 2 TF-IDF        | Lane 1 naive BM25 (reference)   |
|:----------|:---------------------|:---------------------|:--------------------------------|
| Recall@1  | 0.408 [0.358, 0.457] | 0.434 [0.382, 0.483] | 0.373 [0.324, 0.422]            |
| Recall@3  | 0.882 [0.850, 0.913] | 0.870 [0.835, 0.905] | 0.847 [0.809, 0.882]            |
| Recall@5  | 0.942 [0.916, 0.965] | 0.931 [0.902, 0.957] | 0.902 [0.870, 0.934]            |
| Recall@10 | 0.951 [0.928, 0.974] | 0.954 [0.931, 0.974] | 0.934 [0.907, 0.957]            |
| MRR       | 0.639 [0.607, 0.672] | 0.649 [0.616, 0.683] | 0.607 [0.572, 0.639]            |

## Paired bootstrap significance tests

| Comparison | mean A | mean B | diff | p | significant (0.05) |
|---|---|---|---|---|---|
| Lane 2 BM25 vs Lane 1 naive BM25 (Recall@1) | 0.408 | 0.373 | +0.035 | 0.072 | no |
| Lane 2 BM25 vs Lane 1 naive BM25 (Recall@5) | 0.942 | 0.902 | +0.040 | <0.001 | yes |
| Lane 2 BM25 vs Lane 1 naive BM25 (MRR) | 0.639 | 0.607 | +0.033 | 0.006 | yes |
| Lane 2 BM25 vs Lane 2 TF-IDF (Recall@1) | 0.408 | 0.434 | -0.026 | 0.098 | no |
| Lane 2 BM25 vs Lane 2 TF-IDF (Recall@5) | 0.942 | 0.931 | +0.012 | 0.022 | yes |
| Lane 2 BM25 vs Lane 2 TF-IDF (MRR) | 0.639 | 0.649 | -0.009 | 0.328 | no |

## Parameter check on the dev split

80 settings tried (stopwords on/off x stemming on/off x k1 in {0.5..2.0} x b in {0.3..1.0}).
- Submitted default (stopwords + stemming, k1=1.2, b=0.75): dev MRR 0.672, Recall@1 0.449, Recall@5 0.952
- Best setting (stopwords=False, stemming=True, k1=0.5, b=0.3): dev MRR 0.683, Recall@1 0.458, Recall@5 0.958
- Gap: +0.011 MRR, so the standard defaults were kept (no test-set tuning).

## Where BM25 goes wrong (test split)

- Gold evidence at rank 2: 120 / 346 claims. In 55% of these, the rank-1 sentence shares at least 70% of its words with the gold sentence: a near-identical contrastive twin (FEVER-Symmetric pairs often differ by one or two words, e.g. 'English-language' vs 'Spanish-language'). Bag-of-words scoring cannot tell which twin the claim agrees with.
- Gold evidence not in top-10: 17 / 346 claims.

Per-claim ranks: `per_claim_test.csv`. Chart: `lane2_retrieval_ci.png`.