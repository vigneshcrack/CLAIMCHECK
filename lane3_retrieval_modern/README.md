# Lane 3 — Modern Retrieval (Sentence-BERT + FAISS + cross-encoder reranking)

Owner: Nandeeswari

## Description
This folder is ClaimCheck's **semantic evidence retrieval**. Given a claim, it embeds it with Sentence-BERT, finds the nearest evidence sentences in a FAISS index over Lane 1's corpus, and (optionally) re-orders the candidates with a cross-encoder. It returns `evidence_id`s from the same `evidence_corpus.csv` as Lane 2, so Lane 4 and Lane 5 can use either retriever interchangeably.

Shared interface contract, implemented exactly:

```python
from sbert_faiss import retrieve          # (retrieval.py re-exports the same function)
from reranker import rerank

retrieve(claim, k)                       # -> ['ev_334', 'ev_333', ...]  top-k evidence_ids, best-first
rerank(claim, candidates, top_n)         # -> candidates re-ordered by the cross-encoder, cut to top_n
```

Extras, same shapes as Lane 2 so the app can swap retrievers with one line:

```python
retrieve_with_scores(claim, k)           # [(evidence_id, cosine_score, readable_text), ...]
retrieve_batch(claims, k)                # [[evidence_id, ...], ...]  (fast, for evaluation)
rerank_with_scores(claim, candidates, top_n)   # [(evidence_id, rerank_score_0_to_1, readable_text), ...]
search(claim, top_n=5, first_stage="sbert", pool_k=POOL_K)
    # one call for the app: retrieve pool_k -> rerank -> top_n
    # -> [{"evidence_id", "text", "rank", "rerank_score", "retrieval_score", "retrieval_rank"}, ...]
    # first_stage="bm25" reranks Lane 2's BM25 candidates instead
```

## Design
| Choice | Value | Why |
|---|---|---|
| Bi-encoder | `sentence-transformers/all-mpnet-base-v2` | Strongest general-purpose checkpoint in the sentence-transformers benchmark for semantic search; 768-d. Slower than MiniLM (≈5×) but the corpus is only 648 sentences, so quality matters more than speed. `--compare-checkpoints` checks it against `all-MiniLM-L6-v2` and `multi-qa-mpnet-base-cos-v1` on **dev**. |
| Text cleaning | FEVER escapes removed before embedding (`-LRB-` → `(`) | A transformer reads `-LRB-` as junk subwords; BM25's tokenizer strips it anyway. |
| FAISS index | `IndexFlatIP` over L2-normalised vectors | Inner product of unit vectors = cosine similarity. Exact search: approximate indexes (IVF/HNSW) only pay off at ~100k+ vectors and would lose recall. |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reads claim and evidence *together* (full cross-attention), so it can weigh the one or two words that separate FEVER-Symmetric "twin" sentences — exactly where BM25 fails (Lane 2 found the gold sentence at rank 2 for 120/346 test claims). 6 layers: fast enough to rerank 20–50 candidates per claim on CPU. |
| Rerank depth | `POOL_K` candidates in → top-n out | Chosen on the **dev** split from {5, 10, 20, 50} by MRR (`results/dev_pool_sweep.csv`). Test split never used for tuning. |

`rerank_score` is a **relevance** score (sigmoid of the cross-encoder logit), not the probability that the claim is true. Verdicts are Lane 4's job.

## Results (test split, n = 346 claims with gold evidence; 95% bootstrap CI)
| Metric | Lane 2 BM25 | Lane 3 SBERT | **Lane 3 SBERT + rerank** | BM25 + Lane 3 rerank |
|---|---|---|---|---|
| Recall@1 | 0.410 [0.358, 0.460] | 0.405 [0.355, 0.457] | **0.431** [0.379, 0.480] | 0.428 [0.376, 0.477] |
| Recall@3 | 0.887 [0.853, 0.919] | 0.902 [0.867, 0.931] | **0.936** [0.910, 0.960] | 0.934 [0.908, 0.960] |
| Recall@5 | 0.942 [0.916, 0.965] | 0.957 [0.934, 0.977] | **0.983** [0.968, 0.994] | 0.962 [0.942, 0.980] |
| Recall@10 | 0.951 [0.928, 0.971] | 0.980 [0.965, 0.994] | **0.997** [0.991, 1.000] | 0.962 [0.942, 0.980] |
| MRR | 0.640 [0.608, 0.673] | 0.655 [0.624, 0.686] | **0.684** [0.653, 0.713] | 0.675 [0.642, 0.705] |

- **RQ1 — SBERT alone vs BM25: no significant difference** (Recall@5 +0.014, p = 0.33; MRR +0.015, p = 0.35). FEVER-Symmetric claims reuse most of their evidence's wording, so keyword matching is already strong.
- **SBERT + rerank vs BM25: significantly better** on Recall@5 (+0.040, p < 0.001) and MRR (+0.043, p = 0.001). Recall@1 is not significant (p = 0.37).
- **The reranker is what helps:** it significantly improves both SBERT (MRR +0.028, p = 0.022) and BM25 (MRR +0.034, p = 0.011).
- **Recall@1 stays ≈ 0.43 for every system** because of contrastive twins (e.g. Tilda Swinton "Scottish" vs "British" actress): both twins are equally *relevant* to the claim — one supports it, one refutes it — so choosing between them is verification (Lane 4), not retrieval. This is why the app passes the top-5 to Lane 4, where SBERT + rerank finds the gold sentence 98.3% of the time.
- **Dev choices:** rerank pool depth 50 (dev MRR 0.712 vs 0.710 at 20). Checkpoint check on dev: `multi-qa-mpnet-base-cos-v1` scored slightly higher than `all-mpnet-base-v2` (MRR 0.696 vs 0.671, no rerank); kept all-mpnet as the pre-registered choice — noted as a limitation / future work.
- **Speed (Colab, per claim):** BM25 1.8 ms, SBERT 46 ms, cross-encoder over 50 candidates ≈ 1.3 s.

Full tables and tests: `results/lane3_results.md`. Chart: `results/lane3_vs_lane2_ci.png`. 5-claim side-by-side: `results/sample_5_claims.md`.

## How to Setup
- **Python:** 3.10–3.12
- **Packages:** `requirements.txt` in this folder (`sentence-transformers`, `faiss-cpu`, `torch`, …; `rank_bm25` + `nltk` for the Lane 2 comparison)
- **Models:** downloaded from Hugging Face on first use (~420 MB mpnet + ~90 MB cross-encoder), then cached.
- **Data:** none to download. Reads `../lane1_data_eval/data/processed/evidence_corpus.csv` and `claimcheck_dataset.csv`.

## How to Run
From inside this folder:
```bash
pip install -r requirements.txt
python -m pytest tests/ -v              # offline contract tests (fake models, no download, ~2 s)
python build_index.py                   # SBERT embeddings + FAISS index -> indexes/ (~1 min CPU)
python sample_claims.py                 # Lane 2's 5 sample claims, side by side -> results/sample_5_claims.md
python evaluate_lane3.py --compare-checkpoints   # full scoring vs Lane 2 -> results/ (~5–10 min CPU, ~1 min GPU)
```
On Google Colab (free GPU): upload the repo zip, then
```
!unzip -q CLAIMCHECK-main.zip && cd CLAIMCHECK-main/lane3_retrieval_modern && pip install -q -r requirements.txt && python build_index.py && python sample_claims.py && python evaluate_lane3.py --compare-checkpoints
```
`POOL_K` in `reranker.py` is set to 50, the depth chosen on dev, so the app matches the evaluated setting.

## Project Structure
| File | Purpose |
|---|---|
| `sbert_faiss.py` | `retrieve(claim, k)` (shared contract), SBERT encoding, FAISS index build/save/load, auto-rebuild if the corpus changes |
| `retrieval.py` | Alias of `sbert_faiss.py` (file name used in the lane brief) |
| `reranker.py` | `rerank(claim, candidates, top_n)` (shared contract) + `search()` one-call pipeline for the app |
| `build_index.py` | Index build script |
| `sample_claims.py` | 5-claim sanity check on the same IDs as Lane 2 |
| `evaluate_lane3.py` | Scores Lane 3 vs Lane 2 with Lane 1's `evaluation.py`; exports retrieved evidence for the ablation |
| `tests/test_interface.py` | Offline contract tests |
| `results/` | `lane3_results.md/.csv`, `lane3_vs_lane2_ci.png`, `per_claim_test.csv`, `retrieved_test_top10.csv`, `dev_pool_sweep.csv`, `dev_checkpoint_comparison.csv`, `sample_5_claims.md/.csv` |
| `indexes/` | Built automatically, not committed |

## For Lane 1 / Lane 4 — ablation tiers
`results/retrieved_test_top10.csv` has the top-10 evidence for **every** test claim (including NEI), for systems `sbert` and `sbert_rerank`: columns `claim_id, claim, label, gold_evidence_id, system, rank, evidence_id, is_gold, text`. Top-1 tier = `rank == 1`; top-5 tier = `rank <= 5`.

## Citations
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP 2019*, 3982–3992.
- Song, K., Tan, X., Qin, T., Lu, J., & Liu, T.-Y. (2020). MPNet: Masked and permuted pre-training for language understanding. *NeurIPS 2020*.
- Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data, 7*(3), 535–547. (FAISS)
- Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. *arXiv:1901.04085*.
- Bajaj, P., et al. (2016). MS MARCO: A human generated machine reading comprehension dataset. *arXiv:1611.09268*.
- Thorne, J., et al. (2018). FEVER: A large-scale dataset for Fact Extraction and VERification. *NAACL-HLT 2018*, 809–819.
- Schuster, T., et al. (2019). Towards debiasing fact verification models. *EMNLP-IJCNLP 2019*, 3419–3425. (FEVER-Symmetric)
