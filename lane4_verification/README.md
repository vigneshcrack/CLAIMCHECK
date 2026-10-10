# Lane 4 — Verification / NLI

Owner: _(assign teammate name here)_

## What you're building

A `verify(claim, evidence)` function that predicts SUPPORTS / REFUTES / NOT ENOUGH INFO, scored against the shared evaluation harness in `../lane1_data_eval/scripts/evaluation.py`.

## Shared interface contract — implement exactly this signature

```python
def verify(claim: str, evidence: str) -> dict:
    """Return {"label": "SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO", "confidence": float}."""
```

## Data you need (already built by Lane 1 — don't rebuild it)

- `../lane1_data_eval/data/processed/claimcheck_dataset.csv` — has `claim`, `evidence`, `label`, `split` columns ready to train/evaluate on.

## Important — read this before you start

Lane 1 already ran a **lexical-overlap bias analysis** (`../lane1_data_eval/scripts/bias_analysis.py`, write-up in `../lane1_data_eval/reports/bias_analysis_report.md`) and found that a trivial single-feature classifier using ONLY claim-evidence word overlap reaches 56.9% accuracy vs a 45.1% majority baseline (statistically significant, p ≈ 5×10⁻²¹). **Your real NLI model needs to clearly beat 56.9% to demonstrate it's doing genuine entailment reasoning and not just the same lexical shortcut.** Report your accuracy against this baseline explicitly, not just against the majority baseline — it's a stronger, more defensible comparison.

## Steps

1. Pick ONE NLI/verification checkpoint (e.g. a RoBERTa-large MNLI model or DeBERTa-v3 NLI). Justify: model size/speed, relevance of its pretraining to entailment.
2. If time allows, lightly fine-tune on the FEVER-symmetric training split; if not, implement and justify a clear label-mapping layer (entailment→SUPPORTS, contradiction→REFUTES, neutral→NOT ENOUGH INFO) as a documented scope decision.
3. Implement `verify(claim, evidence)` matching the signature above.
4. Run first on GOLD evidence (upper-bound sanity check), then on Lanes 2/3's retrieved evidence once ready.
5. Implement a traditional baseline (TF-IDF features + SVM or Logistic Regression) on the same claim+evidence pairs, for the required traditional-vs-modern comparison.
6. Score yourself with Lane 1's harness: `accuracy`, `macro_f1`, `confusion_matrix_df`, and `bootstrap_ci` for a confidence interval on your accuracy (import pattern: see Lane 2's README).
7. Write a methods paragraph: model choice + justification, label-mapping approach, baseline description, and how you compared against the lexical-overlap baseline above.

## Submit to the group

- `nli_model.py` with `verify(claim, evidence)`
- `traditional_baseline.py`
- Prediction outputs (CSV: claim_id, predicted_label, confidence, gold_label) for both models, for Lane 1 to score
- Methods paragraph for the report

## Commit discipline

Push to this folder daily, even small progress — commit history is evidence of individual contribution for SPARKPlus and the report's "contributions by team member" section.
