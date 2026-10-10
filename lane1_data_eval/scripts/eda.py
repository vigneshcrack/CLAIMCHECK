"""
ClaimCheck — exploratory data analysis (Lane 1)

Reads the processed dataset (produced by load_data.py) and writes a set of
figures to figures/ for use in the Part B report: label distribution, split
distribution, claim length distribution, evidence length distribution, and
claim-length-by-label (a first look at whether length alone separates the
classes — a weaker, more innocuous cousin of the lexical-overlap shortcut
analysis in bias_analysis.py). Also writes a markdown summary to reports/
for direct use when drafting the report's Datasets & EDA section.

How to run (from the project root):
    python3 scripts/eda.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR, get_logger  # noqa: E402

log = get_logger("eda")

PALETTE = {"SUPPORTS": "#2f8f6a", "REFUTES": "#b2483a", "NOT ENOUGH INFO": "#2f6690"}


def load() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "claimcheck_dataset.csv")
    df["claim_len_words"] = df["claim"].astype(str).str.split().apply(len)
    df["evidence_len_words"] = df["evidence"].fillna("").astype(str).str.split().apply(len)
    return df


def plot_label_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [PALETTE.get(lbl, "#888888") for lbl in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    ax.set_title("Label distribution (n={})".format(len(df)))
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "label_distribution.png", dpi=160)
    plt.close(fig)


def plot_split_distribution(df: pd.DataFrame) -> None:
    counts = df["split"].value_counts().reindex(["train", "dev", "test"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(counts.index, counts.values, color="#1f3a52")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    ax.set_title("Split distribution (n={})".format(len(df)))
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "split_distribution.png", dpi=160)
    plt.close(fig)


def plot_claim_length_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(df["claim_len_words"], bins=range(0, 41, 2), color="#2f6690", edgecolor="white")
    ax.set_title("Claim length (words)")
    ax.set_xlabel("words per claim")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "claim_length_distribution.png", dpi=160)
    plt.close(fig)


def plot_evidence_length_hist(df: pd.DataFrame) -> None:
    real = df.loc[df["source"] == "fever_symmetric", "evidence_len_words"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(real, bins=range(0, 81, 4), color="#d98a3d", edgecolor="white")
    ax.set_title("Gold evidence length (words), real rows only")
    ax.set_xlabel("words per evidence sentence")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "evidence_length_distribution.png", dpi=160)
    plt.close(fig)


def plot_claim_length_by_label(df: pd.DataFrame) -> None:
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    data = [df.loc[df["label"] == lbl, "claim_len_words"].values for lbl in labels]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(PALETTE[lbl])
        patch.set_alpha(0.75)
    ax.set_title("Claim length (words) by label")
    ax.set_ylabel("words per claim")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "claim_length_by_label.png", dpi=160)
    plt.close(fig)


def summary_stats(df: pd.DataFrame) -> dict:
    real_ev = df.loc[df["source"] == "fever_symmetric", "evidence_len_words"]
    return {
        "total_rows": len(df),
        "real_rows": int((df["source"] == "fever_symmetric").sum()),
        "synthetic_nei_rows": int((df["source"] == "synthetic_nei_placeholder").sum()),
        "label_counts": df["label"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "claim_len_describe": df["claim_len_words"].describe().to_dict(),
        "evidence_len_describe": real_ev.describe().to_dict(),
        "duplicate_claims": int(df["claim"].duplicated().sum()),
    }


def write_report(stats: dict) -> None:
    cld = stats["claim_len_describe"]
    eld = stats["evidence_len_describe"]
    lines = [
        "# EDA summary — ClaimCheck (Lane 1)",
        "",
        f"Total rows: {stats['total_rows']} ({stats['real_rows']} real FeverSymmetric rows + "
        f"{stats['synthetic_nei_rows']} flagged synthetic NEI placeholder rows — see data provenance",
        "note in `scripts/load_data.py`).",
        "",
        "## Label distribution",
        *[f"- {lbl}: {cnt}" for lbl, cnt in stats["label_counts"].items()],
        "",
        "## Split distribution",
        *[f"- {split}: {cnt}" for split, cnt in stats["split_counts"].items()],
        "",
        "## Claim length (words)",
        f"- mean={cld['mean']:.2f}, std={cld['std']:.2f}, min={cld['min']:.0f}, "
        f"median={cld['50%']:.0f}, max={cld['max']:.0f}",
        "",
        "## Gold evidence length (words), real rows only",
        f"- mean={eld['mean']:.2f}, std={eld['std']:.2f}, min={eld['min']:.0f}, "
        f"median={eld['50%']:.0f}, max={eld['max']:.0f}",
        "",
        f"## Duplicate claims: {stats['duplicate_claims']}",
        "Expected and not a data-cleaning error — FeverSymmetric is built from paraphrase/negation",
        "pairs of the same underlying FEVER claim (one SUPPORTS version, one REFUTES version), which",
        "is itself the dataset's whole point (see `scripts/bias_analysis.py`).",
        "",
        "## Figures",
        "`label_distribution.png`, `split_distribution.png`, `claim_length_distribution.png`, "
        "`evidence_length_distribution.png`, `claim_length_by_label.png` — all in `figures/`.",
    ]
    out_path = REPORTS_DIR / "eda_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    dataset = load()
    stats = summary_stats(dataset)
    log.info("EDA summary: %s", {k: v for k, v in stats.items() if not isinstance(v, dict)})

    plot_label_distribution(dataset)
    plot_split_distribution(dataset)
    plot_claim_length_hist(dataset)
    plot_evidence_length_hist(dataset)
    plot_claim_length_by_label(dataset)
    write_report(stats)

    print("=== ClaimCheck EDA summary ===")
    print(f"Total rows: {stats['total_rows']}")
    print(f"Real (fever_symmetric) rows: {stats['real_rows']}")
    print(f"Synthetic NEI placeholder rows: {stats['synthetic_nei_rows']}")
    print()
    print("Label distribution:", stats["label_counts"])
    print("Split distribution:", stats["split_counts"])
    print()
    print(f"Claim length (words) — mean={stats['claim_len_describe']['mean']:.2f}, "
          f"median={stats['claim_len_describe']['50%']:.0f}")
    print(f"Evidence length (words, real only) — mean={stats['evidence_len_describe']['mean']:.2f}, "
          f"median={stats['evidence_len_describe']['50%']:.0f}")
    print(f"Duplicate claims: {stats['duplicate_claims']}")
    print(f"\nFigures written to {FIGURES_DIR}")
    print(f"Summary report written to {REPORTS_DIR / 'eda_report.md'}")
