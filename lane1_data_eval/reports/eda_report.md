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