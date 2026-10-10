"""
ClaimCheck — Lane 1 pipeline orchestrator.

Runs the full Lane 1 build in one command, in the correct order, with a
single timestamped log for the whole run: load data -> EDA -> evaluation
harness demo (TF-IDF vs BM25 with significance testing) -> lexical-overlap
bias analysis. Each stage writes its own report under reports/ and its own
figures under figures/; this script just sequences them and gives a single
pass/fail summary, which is what you'd want for "did my pipeline actually
run end-to-end" rather than re-reading four separate script outputs.

How to run (from the project root, wherever you've unzipped it):
    python3 -I run_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT, get_logger  # noqa: E402

log = get_logger("run_pipeline")

STAGES = [
    ("Data loading & preprocessing", "scripts/load_data.py"),
    ("Exploratory data analysis", "scripts/eda.py"),
    ("Evaluation harness (TF-IDF vs BM25 + significance testing)", "scripts/evaluation.py"),
    ("Lexical-overlap bias analysis", "scripts/bias_analysis.py"),
    ("Master report assembly", "scripts/build_master_report.py"),
]


def run_stage(name: str, script: str) -> bool:
    log.info("=" * 70)
    log.info("STAGE: %s (%s)", name, script)
    log.info("=" * 70)
    start = time.time()
    result = subprocess.run([sys.executable, script], cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    if result.returncode == 0:
        log.info("Stage '%s' completed in %.1fs", name, elapsed)
        return True
    log.error("Stage '%s' FAILED (exit code %d) after %.1fs", name, result.returncode, elapsed)
    return False


if __name__ == "__main__":
    log.info("Starting ClaimCheck Lane 1 pipeline (project root: %s)", PROJECT_ROOT)
    all_ok = True
    for name, script in STAGES:
        ok = run_stage(name, script)
        all_ok = all_ok and ok
        if not ok:
            log.error("Stopping pipeline early because stage '%s' failed.", name)
            break

    log.info("=" * 70)
    if all_ok:
        log.info("Pipeline completed successfully. Outputs: data/processed/, figures/, reports/, logs/")
        sys.exit(0)
    else:
        log.error("Pipeline did not complete successfully — see log above for the failing stage.")
        sys.exit(1)
