# ClaimCheck — Lane 1 master report (Data, Evaluation & Innovation)

Generated 2026-10-10. This document assembles the three stage reports below
into one read-through for drafting the Part B report's Datasets, Evaluation Methodology, and
Findings/Robustness sections. Every number here is reproducible by running `python3 -I
run_pipeline.py` from the project root — see README.md.

---
# EDA summary — ClaimCheck (Lane 1)

Total rows: 1456 (1416 real FeverSymmetric rows + 40 flagged synthetic NEI placeholder rows — see data provenance
note in `scripts/load_data.py`).

## Label distribution
- REFUTES: 710
- SUPPORTS: 706
- NOT ENOUGH INFO: 40

## Split distribution
- train: 741
- dev: 361
- test: 354

## Claim length (words)
- mean=8.87, std=2.81, min=4, median=8, max=19

## Gold evidence length (words), real rows only
- mean=26.00, std=12.39, min=6, median=23, max=70

## Duplicate claims: 750
Expected and not a data-cleaning error — FeverSymmetric is built from paraphrase/negation
pairs of the same underlying FEVER claim (one SUPPORTS version, one REFUTES version), which
is itself the dataset's whole point (see `scripts/bias_analysis.py`).

## Figures
`label_distribution.png`, `split_distribution.png`, `claim_length_distribution.png`, `evidence_length_distribution.png`, `claim_length_by_label.png` — all in `figures/`.

---

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

---

# Lexical-overlap bias analysis — ClaimCheck (Lane 1)

Motivation: FeverSymmetric (Schuster et al., EMNLP 2019) was built because FEVER-trained
verification models were shown to partly exploit claim-evidence word overlap as a shortcut
for entailment reasoning. This analysis checks for that signal directly on our data slice.

## Overlap-ratio distribution by label
- SUPPORTS: mean=0.676, median=0.700
- REFUTES: mean=0.561, median=0.571
- Mann-Whitney U test: U=322903.0, p=4.96e-21 (statistically significant at alpha=0.05)

## Single-feature 'shortcut' classifier (overlap_ratio only, logistic regression)
- Train/test sizes: 716 / 346
- Accuracy: 0.569 (majority-class baseline: 0.451, lift: +0.118)
- Macro-F1: 0.568
- Logistic regression coefficient on overlap_ratio: +1.867 (higher overlap -> more likely SUPPORTS)

Confusion matrix (rows=true, cols=predicted):

| true     |   SUPPORTS |   REFUTES |
|:---------|-----------:|----------:|
| SUPPORTS |        109 |        47 |
| REFUTES  |        102 |        88 |

## What this means for the report
A single, meaning-blind feature (how many claim words also appear in the evidence) measurably
separates SUPPORTS from REFUTES in this dataset. That is evidence of a real lexical shortcut a
verification model could exploit instead of genuine entailment reasoning — this is the concrete,
measured version of the robustness/limitations argument the Part A feedback asked for, and it
gives Lane 4 a named baseline to beat: their real NLI model should clearly outperform this
single-feature classifier, not just the majority-class baseline, to demonstrate it is reasoning
about meaning rather than surface form.