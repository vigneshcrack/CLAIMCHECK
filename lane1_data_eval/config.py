"""
ClaimCheck — Lane 1 shared configuration and logging setup.

Every script in this project imports its paths and its logger from here,
instead of each hardcoding its own. This is what lets the whole project be
unzipped and run from any location (paths resolve relative to this file),
and gives every run a consistent, timestamped log in logs/ alongside the
usual console output — standard practice for a reproducible research
pipeline, and useful evidence of exactly what ran when, for the report's
"reproducibility" section.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Resolved relative to this file, so the project folder can be moved/unzipped
# anywhere (a different drive, a different username, a teammate's laptop)
# without editing a single path by hand.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

for _d in (PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SPLIT_FILES = {
    "train": "fever_symmetric_v01.jsonl",
    "dev": "fever_symmetric_dev.jsonl",
    "test": "fever_symmetric_test.jsonl",
}

RANDOM_SEED = 42
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
BOOTSTRAP_RESAMPLES = 2000
CONFIDENCE_LEVEL = 0.95


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to both the console and a per-run log
    file under logs/, so every pipeline run leaves a timestamped audit trail
    (what ran, what parameters, what the headline numbers were) alongside
    the usual terminal output.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers if called more than once
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    run_stamp = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(LOGS_DIR / f"pipeline_{run_stamp}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
