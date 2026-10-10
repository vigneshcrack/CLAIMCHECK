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