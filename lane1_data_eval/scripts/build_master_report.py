"""
ClaimCheck — Lane 1 master report assembler.

Stitches the three per-stage reports (EDA, evaluation harness, bias
analysis) plus the dataset provenance note into one polished,
report-ready document: reports/lane1_master_report.md. This is the
document meant to be read end-to-end and largely copy-adapted into the
Part B report's Datasets / Evaluation Methodology / Findings sections,
rather than making the reader piece together four separate files.

Run this LAST, after load_data.py, eda.py, evaluation.py and
bias_analysis.py have all produced their outputs (run_pipeline.py does
this for you automatically).

How to run (from the project root):
    python3 scripts/build_master_report.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR, get_logger  # noqa: E402

log = get_logger("build_master_report")


def _read(name: str) -> str:
    path = REPORTS_DIR / name
    if not path.exists():
        return f"_[{name} not found — run the corresponding script first]_"
    return path.read_text(encoding="utf-8")


def build() -> str:
    eda = _read("eda_report.md")
    evaluation = _read("evaluation_report.md")
    bias = _read("bias_analysis_report.md")

    header = [
        "# ClaimCheck — Lane 1 master report (Data, Evaluation & Innovation)",
        "",
        f"Generated {date.today().isoformat()}. This document assembles the three stage reports below",
        "into one read-through for drafting the Part B report's Datasets, Evaluation Methodology, and",
        "Findings/Robustness sections. Every number here is reproducible by running `python3 -I",
        "run_pipeline.py` from the project root — see README.md.",
        "",
        "---",
        "",
    ]
    body = "\n".join(header) + "\n\n".join([eda, "---", evaluation, "---", bias])
    return body


if __name__ == "__main__":
    content = build()
    out_path = REPORTS_DIR / "lane1_master_report.md"
    out_path.write_text(content, encoding="utf-8")
    log.info("Wrote %s (%d characters)", out_path, len(content))
    print(f"Master report written to {out_path}")
