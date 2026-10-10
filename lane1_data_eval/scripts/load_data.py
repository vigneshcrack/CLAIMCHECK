"""
ClaimCheck — data loading & preprocessing (Lane 1)

IMPORTANT — data provenance note (read this before using this module):

The full FEVER dataset (fever.ai) and the Hugging Face `fever` mirror are NOT
reachable from this build environment (the org's network policy blocks
fever.ai, huggingface.co and s3.amazonaws.com outbound). What IS used here is
FeverSymmetric (Schuster et al., EMNLP 2019, "Towards Debiasing Fact
Verification Models"), a real, published, peer-reviewed derivative of FEVER:
each row is a genuine FEVER claim paired with its real evidence sentence and
real SUPPORTS/REFUTES label. It is downloaded from the authors' own GitHub
repo (TalSchuster/FeverSymmetric), which IS reachable from this environment.

This choice of dataset is not incidental to this module either: FeverSymmetric
was built specifically to test whether verification models exploit lexical
overlap between claim and evidence as a shortcut for genuine entailment
reasoning (every claim appears twice, paired with evidence that supports it
once and refutes it once, holding surface wording roughly constant). That is
exactly what `scripts/bias_analysis.py` quantifies in this project — the data
source and the analysis were chosen together, not independently.

What this sample does NOT give us, and what the full FEVER dataset would add:
  1. NOT ENOUGH INFO (NEI) labelled examples — FeverSymmetric is a 2-class
     (SUPPORTS/REFUTES) bias-evaluation set by design. A small number of
     clearly-flagged SYNTHETIC "no relevant evidence found" rows are added
     below purely so the NEI code path can be exercised end-to-end; they are
     tagged source="synthetic_nei_placeholder" and must not be reported as
     real FEVER NEI examples in the report.
  2. The full Wikipedia evidence corpus (5.4M pages) for open retrieval — we
     instead build the retrieval corpus from the ~650 unique evidence
     sentences actually present across this sample (a standard simplification
     for a scoped coursework retrieval task: retrieve the right sentence out
     of a few hundred candidates rather than all of Wikipedia).

Swapping in the real FEVER train/dev/test + wiki-pages later (e.g. downloaded
from fever.ai on an unrestricted machine, or via `datasets.load_dataset
("fever", "v1.0")` on Colab) requires no code changes elsewhere in the
pipeline — just point `load_raw()` at the new files and keep the same output
schema documented below.

Output schema (pandas DataFrame), after `build_dataset()`:
    claim_id      : str
    claim         : str
    label         : one of {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
    evidence      : str   (gold evidence sentence; empty string for NEI rows)
    split         : one of {"train", "dev", "test"}
    source        : "fever_symmetric" | "synthetic_nei_placeholder"

Security note: this module reads downloaded data files. It is kept in its
own `scripts/` directory, separate from `data/raw/`. If re-downloading raw
files yourself, run this with `python3 -I` (isolated mode) to be safe.

How to run (from the project root, wherever you've unzipped it):
    python3 -I scripts/load_data.py
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, PROCESSED_DIR, SPLIT_FILES, RANDOM_SEED, get_logger  # noqa: E402

log = get_logger("load_data")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_raw() -> pd.DataFrame:
    """Load and normalise the three FeverSymmetric splits into one schema.

    The three files don't share identical column names (v0.1 uses
    `evidence_sentence`; v0.2 dev/test use `evidence`/`gold_label`), so this
    normalises them before anything downstream has to care.
    """
    all_rows = []
    for split, fname in SPLIT_FILES.items():
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Expected raw file not found: {path}. "
                "Check you're running from the unzipped project root, and that "
                "data/raw/ contains the 3 FeverSymmetric JSONL files."
            )
        raw_rows = _read_jsonl(path)
        log.info("Loaded %d raw rows from %s (split=%s)", len(raw_rows), fname, split)
        for r in raw_rows:
            claim = r.get("claim", "").strip()
            evidence = (r.get("evidence_sentence") or r.get("evidence") or "").strip()
            label = (r.get("label") or r.get("gold_label") or "").strip().upper()
            if label not in {"SUPPORTS", "REFUTES"}:
                continue  # skip anything malformed rather than silently mislabel
            all_rows.append(
                {
                    "claim_id": f"{split}_{r.get('id')}",
                    "claim": claim,
                    "label": label,
                    "evidence": evidence,
                    "split": split,
                    "source": "fever_symmetric",
                }
            )
    df = pd.DataFrame(all_rows)
    before = len(df)
    df = df.drop_duplicates(subset=["claim", "evidence", "label"]).reset_index(drop=True)
    log.info("Dropped %d exact duplicate (claim, evidence, label) rows", before - len(df))
    return df


def add_synthetic_nei(df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """Add a small, clearly-flagged set of synthetic NOT ENOUGH INFO rows.

    Built by pairing a claim with a *mismatched* evidence sentence drawn from
    elsewhere in the corpus (so the NLI/verification stage sees a claim with
    no genuinely relevant evidence attached, as a real NEI case would look at
    inference time). This is a placeholder for exercising the pipeline's NEI
    branch only — it is NOT real annotated FEVER data and must be reported as
    such.
    """
    rng = random.Random(RANDOM_SEED)
    claims_pool = df[["claim_id", "claim", "split"]].drop_duplicates("claim_id").to_dict("records")
    evidence_pool = df["evidence"].unique().tolist()

    synthetic_rows = []
    chosen = rng.sample(claims_pool, k=min(n, len(claims_pool)))
    for i, row in enumerate(chosen):
        mismatched_evidence = rng.choice(evidence_pool)
        synthetic_rows.append(
            {
                "claim_id": f"synthetic_nei_{i}",
                "claim": row["claim"],
                "label": "NOT ENOUGH INFO",
                "evidence": "",  # NEI rows carry no gold evidence by definition
                "split": row["split"],
                "source": "synthetic_nei_placeholder",
                "_distractor_evidence_for_testing_only": mismatched_evidence,
            }
        )
    log.info("Added %d synthetic NOT ENOUGH INFO placeholder rows (seed=%d)", len(synthetic_rows), RANDOM_SEED)
    synthetic_df = pd.DataFrame(synthetic_rows)
    combined = pd.concat([df, synthetic_df], ignore_index=True, sort=False)
    return combined


def build_evidence_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Build the retrieval corpus from unique real evidence sentences.

    This is what retrieve(claim, k) in Lanes 2/3 will search over, and the
    gold evidence_id for a claim is the row whose text matches its own
    `evidence` column (empty/NaN for NEI rows, which have no gold evidence).
    """
    real_evidence = df.loc[df["source"] == "fever_symmetric", "evidence"].dropna()
    unique_evidence = sorted(set(e for e in real_evidence if e))
    corpus = pd.DataFrame(
        {
            "evidence_id": [f"ev_{i}" for i in range(len(unique_evidence))],
            "text": unique_evidence,
        }
    )
    return corpus


def build_dataset(include_synthetic_nei: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw()
    corpus = build_evidence_corpus(df)

    evidence_to_id = dict(zip(corpus["text"], corpus["evidence_id"]))
    df["gold_evidence_id"] = df["evidence"].map(evidence_to_id)

    if include_synthetic_nei:
        df = add_synthetic_nei(df)

    return df, corpus


if __name__ == "__main__":
    dataset, corpus = build_dataset()
    dataset.to_csv(PROCESSED_DIR / "claimcheck_dataset.csv", index=False)
    corpus.to_csv(PROCESSED_DIR / "evidence_corpus.csv", index=False)
    log.info("Wrote claimcheck_dataset.csv (%d rows) and evidence_corpus.csv (%d rows) to %s",
              len(dataset), len(corpus), PROCESSED_DIR)
    print(f"dataset rows: {len(dataset)}  |  corpus size: {len(corpus)}")
    print(dataset["label"].value_counts())
    print(dataset["split"].value_counts())
