# Lane 4 — Verification / NLI

Owner: Pranav Shalmraj

## Description

This folder contains the verification system used for ClaimCheck's claim verification. Given a claim and evidence, it predicts SUPPORTS, REFUTES, or NOT ENOUGH INFO for the claim based on the evidence. It contains a Natural Language Inference (NLI) model which is used for the verification, along with a traditional baseline done using TF-IDF and Logistic Regression to compare. There is also a file for retrieval verification, which uses the retrieval methods used in lane 2 and 3 to obtain evidence and verify claims.

## Shared interface contract, implemented exactly

```python
from lane4_verification.nli_model import verify, verify_batch

verify("Telemundo is a Spanish-language network.",
       "Telemundo is an American Spanish-language television network.")
```
Evidence is the NLI premise and the claim is the hypothesis.

## Retrieval to verdict

```python
from lane4_verification.retrieval_verification import verify_retrieved_claim

result = verify_retrieved_claim(
    "Telemundo is a Spanish-language television network.",
    method="sbert_rerank", k=5, strategy="conflict_abstain",
)
result["evidence"]
result["verdict"]
```
Each retrieved sentence is verified separately (one batched NLI call per claim), so the app can show a verdict per evidence sentence. Retrieval scores are not truth probabilities and are not comparable between methods.

## Saved results

**Model comparison** (`model_comparison.csv`)

| Model | Accuracy (95% CI) | Macro-F1 (SUPPORTS/REFUTES) | Recall SUPPORTS | Recall REFUTES | NEI predictions |
|---|---|---|---|---|---|
| Majority (always REFUTES) | 54.9% (49.7–59.8) | 35.4% | 0.0% | 100.0% | 0 |
| TF-IDF + LogReg | 40.8% (35.8–46.0) | 30.5% | 87.8% | 2.1% | 0 |
| NLI, zero-shot DeBERTa-v3 | **78.9%** (74.6–83.2) | 85.4% | 69.2% | 86.8% | 58 |

**Evidence-quality ablation** (`ablation_summary.csv`; top-k evidence is concatenated into one string)

| Evidence | Accuracy (95% CI) | Macro-F1 | NEI predictions | Gold sentence among retrieved |
|---|---|---|---|---|
| Gold | 78.9% (74.6–83.2) | 85.4% | 58 | 100% |
| Top-1 (BM25) | 40.5% (35.3–45.7) | 45.0% | 80 | 40.5% |
| Top-5 (BM25) | 52.3% (47.1–57.5) | 50.9% | 21 | 94.2% |

How to read them:

- With gold evidence, 58 of the 73 NLI errors are NOT ENOUGH INFO predictions. When the model commits to SUPPORTS or REFUTES it is right 273 of 288 times (94.8%).
- Top-5 retrieval contains the gold sentence 94% of the time, yet accuracy falls from 78.9% to 52.3%. So the loss comes mostly from what happens after retrieval, not from missing the gold sentence. A likely cause is near-duplicate sibling sentences with opposite labels landing in the same concatenated string, but this has not been tested.
- The TF-IDF baseline is essentially predicting SUPPORTS for nearly everything (see recalls). Its score reflects a train/test mismatch (see Known limitations), not a meaningful baseline measurement.
- Don't compare the ablation rows to the gold-evidence rows as if every system received the same evidence.

## Aggregation results on dev (five retrieval methods)

`evaluate_aggregation.py --method all --split dev`: 354 real dev claims, top-5 retrieval, each retrieved sentence verified separately. These are **dev** numbers. They are not comparable row-for-row with the test tables above, and test has not been run for these methods. On dev the majority baseline (always REFUTES) is **51.4%** (182 of 354). The prediction and summary CSVs for all five methods are in `results/`.

**`top1`: use the first-ranked sentence's label**

| Method | Accuracy | Macro-F1 (S/R) | NEI predictions | Accuracy when committed | Recall SUPPORTS | Recall REFUTES |
|---|---|---|---|---|---|---|
| bm25 | 41.5% | 46.1% | 91 | 55.9% | 58.1% | 25.8% |
| tfidf | 41.5% | 46.3% | 91 | 55.9% | 57.0% | 26.9% |
| sbert | 40.1% | 43.1% | 93 | 54.4% | 62.2% | 19.2% |
| sbert_rerank | 42.9% | 45.1% | 79 | 55.3% | 66.3% | 20.9% |
| bm25_rerank | 42.9% | 45.1% | 79 | 55.3% | 66.3% | 20.9% |

**`conflict_abstain`: abstain whenever the five sentences disagree**

| Method | Accuracy | Macro-F1 (S/R) | NEI predictions | Abstention rate | Accuracy when committed | Conflict rate |
|---|---|---|---|---|---|---|
| bm25 | 22.0% | 30.6% | 223 | 63.0% | 59.5% | 58.2% |
| tfidf | 22.6% | 31.0% | 217 | 61.3% | 58.4% | 57.3% |
| sbert | 21.8% | 30.2% | 225 | 63.6% | 59.7% | 59.9% |
| sbert_rerank | 21.8% | 30.5% | 225 | 63.6% | 59.7% | 59.6% |
| bm25_rerank | 20.6% | 28.8% | 230 | 65.0% | 58.9% | 60.7% |

"Accuracy when committed" is accuracy over the claims where the strategy did not answer NOT ENOUGH INFO. "Conflict rate" is the share of claims whose top-5 contained both a SUPPORTS and a REFUTES sentence.

### What the results show

1. **Nothing beats the majority baseline.** Every `top1` run (40.1–42.9%) is below always-REFUTES (51.4%), and every `conflict_abstain` run (20.6–22.6%) is far below it. With the gold sentence the same NLI model scores 78.9% on test, so most of the loss comes from the retrieved evidence, not from the verifier. That 78.9% is a test figure; the dev gold-evidence run is not in this folder.
2. **The retrieval methods are not distinguishable.** The `top1` accuracy spread is 2.8 points, while a 95% interval on 354 claims is roughly ±5 points. The rerankers' small lead comes mainly from fewer NOT ENOUGH INFO answers (79 versus 91–93), not from better REFUTES handling (REFUTES recall stays at 19–27%). No paired bootstrap tests were run, so don't claim a winner without them.

**File formats**

- `predictions_nli.csv`, `predictions_tfidf.csv`: `claim_id, predicted_label, confidence, gold_label`
- `predictions_nli_tiers.csv`: `claim_id, tier, retrieved_ids, predicted_label, confidence, gold_label`
- `aggregation_<method>_<split>_top<k>_predictions.csv`, `..._summary.csv`, `aggregation_comparison_<split>_top<k>.csv`: written by `evaluate_aggregation.py`

## Reproducing

**From the repository root**, with the virtual environment active:

```bash
# unit tests (no model download)
python -m pytest lane4_verification/tests/ -v

# aggregation evaluation: quick check on dev first, then full dev, then ONE test run
python -m lane4_verification.evaluate_aggregation --method all --split dev --limit 20
python -m lane4_verification.evaluate_aggregation --method all --split dev
python -m lane4_verification.evaluate_aggregation --method sbert_rerank --split test

# baseline, ablation, comparison table
python lane4_verification/traditional_baseline.py
python lane4_verification/run_ablation.py --split test
python lane4_verification/models_comparison.py
```

If `run_ablation.py` raises `ModuleNotFoundError: lane2_retrieval_traditional`, the repository root is not on the Python path. Prefix the command with `PYTHONPATH=.` or use `python -m lane4_verification.run_ablation`.

**From inside `lane4_verification/`** (these scripts use relative paths):

```bash
python test.py            # regenerates results/predictions_nli.csv
python nei_preds.py       # error-analysis sample of NOT ENOUGH INFO predictions
python explore.py         # dataset counts
python result_analysis.py results/predictions_nli.csv results/predictions_tfidf.csv
```

`traditional_baseline.py`, `run_ablation.py` and `models_comparison.py` anchor their paths to the file location, so they write to `results/` regardless of where you launch them.

## Tests

23 unit tests cover the aggregation rules (conflict abstention, `top1`, input validation) and the retrieval interface (each of the five methods, score filtering, de-duplication, invalid inputs). Models and retrievers are replaced by stubs, so **the tests do not exercise the real DeBERTa, SBERT, FAISS or cross-encoder models** and say nothing about accuracy.

## Design decisions

- **Checkpoint:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (about 184M parameters; runs on a laptop or free Colab GPU). Trained on MNLI, FEVER and ANLI, so the entailment / contradiction / neutral mapping works zero-shot.
- **Scope decision:** zero-shot plus a documented label-mapping layer (entailment to SUPPORTS, contradiction to REFUTES, neutral to NOT ENOUGH INFO), with no fine-tuning. The real training split is small (716 rows) and, as noted below, leaks into the test split.
- **Task framing:** the headline task is **2-class** (SUPPORTS / REFUTES) on the `fever_symmetric` rows. The 40 synthetic NOT ENOUGH INFO rows are excluded from headline numbers, because their empty evidence makes them trivially correct.
- **Baseline:** claim and evidence are vectorised separately (TF-IDF, 1–2 grams, `min_df=2`, sublinear TF) and concatenated, so the model sees both texts but cannot compare them. `C` is chosen on dev. No overlap features, which would rebuild Lane 1's lexical-overlap baseline.
- **Evidence input:** the NLI model takes one string. The legacy helpers join top-k sentences with `EVIDENCE_SEP = " "`; the unified interface avoids joining by verifying each sentence separately. Pairs longer than 512 tokens are truncated silently.

## Known limitations and things to check

1. **Train/test leakage in the current splits.** In Lane 4's checks, 120 of 346 test claims also appear in train, 99% of those twins carry the opposite label, and 144 test rows share evidence with train. Zero-shot NLI results are unaffected, but anything trained on these splits (the TF-IDF baseline, any fine-tuning) is not trustworthy until Lane 1 regroups the splits. Rerun after that.
2. **Majority baseline.** The Lane 4 brief quotes a 45.1% majority baseline and a 56.9% lexical-overlap baseline. On the real test rows the true majority class (REFUTES) is 54.9%; 45.1% is the SUPPORTS share. Ask Lane 1 how the baseline was computed before citing either figure.
3. **FEVER exposure.** The checkpoint was trained on FEVER-derived data and FeverSymmetric derives from FEVER, so accuracy may be partly inflated. A second checkpoint without FEVER (for example `cross-encoder/nli-deberta-v3-base`) has not been run.
4. **NOT ENOUGH INFO is an abstention here.** The real rows have no gold NOT ENOUGH INFO, so every such prediction, including conflict abstentions, counts as an error.
5. **The evidence is partly edited.** FeverSymmetric evidence includes annotator-edited sentences, so a label describes the claim–evidence match, not real-world truth. A model that recalls the true fact can be pulled the wrong way.
6. **Closed, small corpus.** Retrieval over this corpus looks easier than retrieval over Wikipedia, and nothing here verifies arbitrary web claims.
7. **Aggregation is unvalidated** (see above), and `evaluate_aggregation.py` deduplicates and filters retrieved items, so its top-1 can differ from `run_ablation.py`'s top-1.
8. **Reported scores are historical.** Nothing in this folder was rerun for this README; the unit tests passed in a fresh environment.