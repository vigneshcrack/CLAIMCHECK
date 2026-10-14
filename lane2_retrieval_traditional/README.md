# Lane 2 — Traditional Retrieval (BM25 / TF-IDF)

## Description
This folder is ClaimCheck's **traditional (keyword-based) evidence retrieval baseline**. Given a claim, it ranks the sentences in Lane 1's evidence corpus and returns the top-k `evidence_id`s, best first. These go to the reranker (Lane 3) and the NLI verifier (Lane 4), and are shown in the app (Lane 5).

It implements the shared interface contract exactly:

```python
from retrieval import retrieve
retrieve("Tilda Swinton is only Scottish .", k=3)
# -> ['ev_334', 'ev_333', 'ev_632']

retrieve(claim, k, method="tfidf")      # optional: TF-IDF instead of BM25 (default)
retrieve_with_scores(claim, k)          # [(evidence_id, score, readable_text), ...] for the app
```

## Why this design?
- **Okapi BM25 as the traditional baseline:** BM25 is the standard keyword-ranking function in information retrieval and was the retrieval baseline in the original FEVER paper. It needs no training, is fast, and its scores are easy to explain (which words matched). That makes it the natural "traditional" side of RQ1, compared against SBERT in Lane 3.
- **Proper preprocessing instead of `.split()`:** the corpus uses FEVER escape tokens (`-LRB-` for "("), so text is unescaped, lowercased, split into alphanumeric tokens, stopwords are removed, and words are Porter-stemmed. Compared with Lane 1's quick whitespace BM25, this significantly improves Recall@5 and MRR on the test split (see Results).
- **Standard parameters, checked on dev:** k1 = 1.2 and b = 0.75 (the Robertson/Lucene defaults). A sweep of 80 settings on the **dev** split found the best only +0.011 MRR higher, so the defaults were kept and the test split was never used for tuning.
- **Same corpus, same harness:** indexes Lane 1's `evidence_corpus.csv` and is scored only with Lane 1's `scripts/evaluation.py`, so numbers are directly comparable with every other lane.
- **TF-IDF included:** a second classic sparse method (unigram + bigram, sublinear TF, cosine similarity) for comparison.

## Results (test split, n = 346 claims with gold evidence; 95% bootstrap CI)
| Metric | Lane 2 BM25 | Lane 2 TF-IDF | Lane 1 naive BM25 (reference) |
|---|---|---|---|
| Recall@1 | 0.408 [0.358, 0.457] | 0.434 [0.382, 0.483] | 0.373 [0.324, 0.422] |
| Recall@3 | 0.882 [0.850, 0.913] | 0.870 [0.835, 0.905] | 0.847 [0.809, 0.882] |
| Recall@5 | 0.942 [0.916, 0.965] | 0.931 [0.902, 0.957] | 0.902 [0.870, 0.934] |
| Recall@10 | 0.951 [0.928, 0.974] | 0.954 [0.931, 0.974] | 0.934 [0.907, 0.957] |
| MRR | 0.639 [0.607, 0.672] | 0.649 [0.616, 0.683] | 0.607 [0.572, 0.639] |

- **Lane 2 BM25 vs Lane 1 naive BM25:** significantly better on Recall@5 (+0.040, p < 0.001) and MRR (+0.033, p = 0.006). Recall@1 is not significant (p = 0.072).
- **BM25 vs TF-IDF:** BM25 is significantly better on Recall@5 (+0.012, p = 0.022). Recall@1 and MRR are not significantly different.
- Full tables and tests: `results/lane2_results.md`. Chart: `results/lane2_retrieval_ci.png`.

## Sanity check: 5 sample claims (test split, top-5)
| Claim ID | Claim | Label | BM25 gold rank | TF-IDF gold rank |
|---|---|---|---|---|
| test_1255130000003 | The dress inspired millions of hashtags. | SUPPORTS | 2 | 2 |
| test_394050000004 | Lou Gehrig was a baseball player who defended the area near the second base. | SUPPORTS | 1 | 1 |
| test_194469 | Tilda Swinton is only Scottish. | REFUTES | 2 | 2 |
| test_1947750000004 | Larry the Cable Guy series finale aired on the month of August. | REFUTES | 4 | 4 |
| synthetic_nei_9 | Basildon is far away from England. | NOT ENOUGH INFO | no gold evidence | no gold evidence |

Full tables: `results/sample_5_claims.md`. **Lane 3 should run the same IDs** (`results/sample_claim_ids.txt`).

## Challenges we faced
- **Contrastive "twin" sentences:** the dataset contains pairs of evidence sentences that differ by only one or two words (e.g. Tilda Swinton is a "*British*" vs "*Scottish*" actress). BM25 scores both almost equally and often ranks the wrong twin first. The gold sentence is at rank 2 for 120 of 346 test claims, and in 55% of those the rank-1 sentence shares at least 70% of its words with the gold one. This is why Recall@1 (0.41) is much lower than Recall@5 (0.94), and why reranking (Lane 3) matters.
- **Negation and stopwords:** words like "not", "never" and "only" are in the standard stopword list, so BM25 ignores exactly the words that flip a claim's meaning. Keeping them made no measurable difference on dev, because both twins usually still match.
- **Keyword-only matching:** claims that paraphrase the evidence (e.g. "defended the area near second base" vs "second baseman") match only on shared words, so semantic retrieval (SBERT) is expected to help here.

## Future enhancements
- Rerank the BM25 top-10 with a cross-encoder (Lane 3) to separate contrastive twins.
- Keep negation words in the BM25 query and add a small bonus for them.
- Query expansion with synonyms, or a hybrid BM25 + SBERT score.

## How to Setup
- **Python:** 3.10–3.12
- **Packages:** `rank_bm25`, `scikit-learn`, `numpy`, `pandas`, `nltk`, `matplotlib`, `tabulate` (`requirements.txt` in this folder)
- **Data:** none to download. It reads `../lane1_data_eval/data/processed/evidence_corpus.csv` and `claimcheck_dataset.csv`.

## How to Run
From inside this folder:
```bash
pip install -r requirements.txt
python build_index.py        # builds indexes/bm25.pkl + tfidf.pkl (under 1 second)
python sample_claims.py      # 5-claim sanity check -> results/sample_5_claims.md
python evaluate_lane2.py     # full scoring with Lane 1's harness -> results/
```

## Project Structure
| File | Purpose |
|---|---|
| `retrieval.py` | `retrieve(claim, k)` (shared contract), BM25 and TF-IDF indexes, preprocessing |
| `build_index.py` | Index build script |
| `sample_claims.py` | 5-claim sanity check (shared IDs with Lane 3) |
| `evaluate_lane2.py` | Scores Lane 2 with Lane 1's `evaluation.py`: dev sweep, test metrics with CIs, significance tests, chart |
| `requirements.txt` | Dependencies |
| `results/` | `lane2_results.md/.csv`, `lane2_retrieval_ci.png`, `per_claim_test.csv`, `dev_tuning_sweep.csv`, `sample_5_claims.md/.csv`, `sample_claim_ids.txt` |
| `indexes/` | Built automatically and not committed (see `.gitignore`) |

## Citations
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: A large-scale dataset for Fact Extraction and VERification. *Proceedings of NAACL-HLT 2018*, 809–819.
- Schuster, T., Shah, D., Yeo, Y. J. S., Filizzola, D., Santus, E., & Barzilay, R. (2019). Towards debiasing fact verification models. *Proceedings of EMNLP-IJCNLP 2019*, 3419–3425. (FEVER-Symmetric)
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Brown, D. (2020). *rank_bm25* [Computer software]. https://github.com/dorianbrown/rank_bm25
