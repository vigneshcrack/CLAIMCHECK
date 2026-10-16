# Lane 3 — Modern retrieval results (scored with Lane 1's evaluation harness)

SBERT checkpoint: `sentence-transformers/all-mpnet-base-v2` (768-d, FAISS IndexFlatIP, cosine). Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
Corpus: 648 evidence sentences. Test split: 346 claims with a gold evidence id (NEI claims have no gold evidence and are excluded from retrieval metrics, as the harness requires).

## SBERT checkpoint comparison (DEV only, no rerank)

| checkpoint                                       |   dim |   recall_at_1 |   recall_at_5 |   mrr |   seconds_index_plus_queries |
|:-------------------------------------------------|------:|--------------:|--------------:|------:|-----------------------------:|
| sentence-transformers/all-mpnet-base-v2          |   768 |         0.438 |         0.949 | 0.671 |                         10.4 |
| sentence-transformers/all-MiniLM-L6-v2           |   384 |         0.444 |         0.946 | 0.67  |                          4.3 |
| sentence-transformers/multi-qa-mpnet-base-cos-v1 |   768 |         0.463 |         0.972 | 0.696 |                          8.3 |

## Rerank pool depth (chosen on DEV only)

| first_stage   |   pool_depth |   recall_at_1 |   recall_at_5 |   mrr |   pool_recall |
|:--------------|-------------:|--------------:|--------------:|------:|--------------:|
| sbert         |            5 |         0.475 |         0.949 | 0.693 |         0.949 |
| sbert         |           10 |         0.48  |         0.969 | 0.705 |         0.975 |
| sbert         |           20 |         0.483 |         0.98  | 0.71  |         0.986 |
| sbert         |           50 |         0.483 |         0.986 | 0.712 |         0.994 |
| bm25          |            5 |         0.48  |         0.952 | 0.696 |         0.952 |
| bm25          |           10 |         0.48  |         0.966 | 0.703 |         0.969 |
| bm25          |           20 |         0.483 |         0.975 | 0.707 |         0.977 |
| bm25          |           50 |         0.483 |         0.975 | 0.707 |         0.977 |

Chosen pool depth by dev MRR: SBERT -> 50, BM25 -> 20 (ties broken toward the smaller, faster pool). Rerank output is cut to top-10.

## Test split: Recall@k and MRR, 95% bootstrap CI (n=346)

|           | Lane 2 BM25          | Lane 3 SBERT         | Lane 3 SBERT + rerank   | Lane 2 BM25 + Lane 3 rerank   |
|:----------|:---------------------|:---------------------|:------------------------|:------------------------------|
| Recall@1  | 0.408 [0.358, 0.460] | 0.405 [0.355, 0.457] | 0.431 [0.379, 0.480]    | 0.428 [0.376, 0.477]          |
| Recall@3  | 0.882 [0.847, 0.913] | 0.902 [0.867, 0.931] | 0.936 [0.910, 0.960]    | 0.934 [0.908, 0.960]          |
| Recall@5  | 0.942 [0.916, 0.965] | 0.957 [0.934, 0.977] | 0.983 [0.968, 0.994]    | 0.962 [0.942, 0.980]          |
| Recall@10 | 0.951 [0.928, 0.971] | 0.980 [0.965, 0.994] | 0.997 [0.991, 1.000]    | 0.962 [0.942, 0.980]          |
| MRR       | 0.639 [0.606, 0.672] | 0.655 [0.624, 0.686] | 0.684 [0.653, 0.713]    | 0.675 [0.642, 0.705]          |

## Paired bootstrap significance tests (same test claims for both systems)

| Comparison | Metric | A | B | diff (A−B) | p | significant (0.05) |
|---|---|---|---|---|---|---|
| RQ1: SBERT vs BM25 | Recall@1 | 0.405 | 0.408 | -0.003 | 0.975 | no |
| RQ1: SBERT vs BM25 | Recall@5 | 0.957 | 0.942 | +0.014 | 0.330 | no |
| RQ1: SBERT vs BM25 | MRR | 0.655 | 0.639 | +0.016 | 0.296 | no |
| SBERT + rerank vs SBERT | Recall@1 | 0.431 | 0.405 | +0.026 | 0.253 | no |
| SBERT + rerank vs SBERT | Recall@5 | 0.983 | 0.957 | +0.026 | 0.004 | yes |
| SBERT + rerank vs SBERT | MRR | 0.684 | 0.655 | +0.028 | 0.022 | yes |
| SBERT + rerank vs BM25 | Recall@1 | 0.431 | 0.408 | +0.023 | 0.319 | no |
| SBERT + rerank vs BM25 | Recall@5 | 0.983 | 0.942 | +0.040 | <0.001 | yes |
| SBERT + rerank vs BM25 | MRR | 0.684 | 0.639 | +0.044 | 0.002 | yes |
| BM25 + rerank vs BM25 | Recall@1 | 0.428 | 0.408 | +0.020 | 0.389 | no |
| BM25 + rerank vs BM25 | Recall@5 | 0.962 | 0.942 | +0.020 | 0.001 | yes |
| BM25 + rerank vs BM25 | MRR | 0.675 | 0.639 | +0.035 | 0.012 | yes |
| SBERT + rerank vs BM25 + rerank | Recall@1 | 0.431 | 0.428 | +0.003 | 0.731 | no |
| SBERT + rerank vs BM25 + rerank | Recall@5 | 0.983 | 0.962 | +0.020 | 0.001 | yes |
| SBERT + rerank vs BM25 + rerank | MRR | 0.684 | 0.675 | +0.009 | 0.004 | yes |

## Contrastive twins (claims where BM25 ranked the gold sentence 2nd)

120 test claims. Gold moved to rank 1 by: SBERT 28, SBERT + rerank 16, BM25 + rerank 16.

## Speed (machine that ran this; per claim, averaged over the test split; SBERT queries encoded in batches)

- Lane 2 BM25: 1.0 ms
- Lane 3 SBERT: 1.1 ms
- cross-encoder (SBERT pool): 35.7 ms

Files: `per_claim_test.csv` (gold rank per system), `retrieved_test_top10.csv` (top-10 evidence per test claim, incl. NEI, systems `sbert` and `sbert_rerank` — use rank 1 / ranks 1–5 for the ablation tiers), `lane3_vs_lane2_ci.png`, `dev_pool_sweep.csv`, `dev_checkpoint_comparison.csv`.