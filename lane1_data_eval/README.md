# ClaimCheck — Lane 1 (Data, Evaluation & Innovation Lead)

A reproducible data + evaluation pipeline for the ClaimCheck claim-verification project (36118 ANLP, AT2 Part B). This is the shared foundation the rest of the team's retrieval/NLI/app components plug into and get scored against.

## Quick start

```
cd claimcheck-lane1
pip install -r requirements.txt

python3 -I run_pipeline.py      # runs every stage below, in order, with one log
python3 -m pytest tests/ -v     # runs the unit tests for the metrics module
```

`run_pipeline.py` is the one command that proves everything works end-to-end: it loads the raw data, runs EDA, runs the evaluation harness (with two retrieval baselines, confidence intervals, and a significance test), runs the lexical-overlap bias analysis, and assembles the master report. Every run also writes a timestamped log to `logs/`.

You can also run any stage individually:

```
python3 -I scripts/load_data.py       # raw JSONL -> data/processed/*.csv
python3 scripts/eda.py                # -> figures/*.png + reports/eda_report.md
python3 scripts/evaluation.py         # TF-IDF vs BM25 + CIs + significance test -> reports/evaluation_report.md
python3 scripts/bias_analysis.py      # lexical-overlap bias check -> reports/bias_analysis_report.md
python3 scripts/build_master_report.py  # combines the 3 reports above -> reports/lane1_master_report.md
```

## What to actually read

Start with **`reports/lane1_master_report.md`** after running the pipeline once — it's the single document written for the Part B report, combining the data provenance note, EDA findings, baseline evaluation results (with confidence intervals and a significance test), and the bias analysis, in report-ready prose and tables.

## Project structure

```
claimcheck-lane1/
├── config.py                  # single source of truth for paths, seeds, logging
├── run_pipeline.py            # orchestrates every stage, one log per run
├── requirements.txt
├── data/
│   ├── raw/                   # 3 downloaded FeverSymmetric JSONL files
│   └── processed/             # claimcheck_dataset.csv, evidence_corpus.csv
├── scripts/
│   ├── load_data.py           # raw -> clean dataset + retrieval corpus
│   ├── eda.py                 # exploratory analysis + charts
│   ├── evaluation.py          # SHARED metrics module (recall@k, MRR, accuracy,
│   │                          #   macro-F1, confusion matrix, FEVER score,
│   │                          #   bootstrap CIs, paired significance testing)
│   └── bias_analysis.py       # lexical-overlap shortcut analysis (ties directly
│                               #   to why FeverSymmetric was chosen as the dataset)
├── tests/
│   └── test_evaluation.py     # 14 unit tests, hand-computed expected values
├── figures/                   # 7 PNG charts for the report
├── reports/                   # markdown write-ups, one per stage + a master report
└── logs/                      # timestamped run logs
```

## Why this is built this way (for anyone reviewing the code)

- **Everything is reproducible from one command** (`run_pipeline.py`) because the "Code focused — reproducibility" grading criterion rewards exactly that, not a notebook that only runs on one person's machine.
- **`evaluation.py` is the single shared metrics module** every lane uses — nobody hand-rolls their own accuracy calculation, so every number in the final report was produced identically, and it's unit-tested (`tests/`) against hand-computed values so a bug can't silently distort the submission.
- **Every reported number ships with a 95% bootstrap confidence interval**, and comparisons between two systems (e.g. BM25 vs TF-IDF) ship with a paired significance test. This directly answers the Part A feedback's core criticism — that claims were asserted ("established approach") rather than evidenced.
- **The bias analysis isn't generic EDA** — it's the specific, named research question FeverSymmetric was built to test (lexical-overlap shortcuts in verification models), measured directly on our data and turned into a concrete baseline ("beat this simple classifier") for Lane 4's real NLI model.

## Data provenance (read before using in the report)

The full FEVER dataset (fever.ai, Hugging Face) wasn't reachable from the environment this was built in, so **FeverSymmetric** (Schuster et al., EMNLP 2019) — a real, published, peer-reviewed FEVER derivative — was used instead. It's genuine FEVER claims/evidence/labels, just a smaller 2-class (SUPPORTS/REFUTES) slice, with 40 clearly-flagged *synthetic* NOT-ENOUGH-INFO rows added only to exercise that code path. Full details are in the module docstring at the top of `scripts/load_data.py`, and the full reasoning is in the project doc `lane1-progress.md`.
