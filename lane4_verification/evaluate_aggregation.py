"""Evaluate a PROPOSED claim-level rule on held-out examples.

Usage from repo root:
  python -m lane4_verification.evaluate_aggregation --method bm25 --split test --limit 20
  python -m lane4_verification.evaluate_aggregation --method sbert --split test

Requires Lane 1 processed data, relevant retrieval modules and cached NLI model.
This runs model inference; existing reported scores are NOT replaced.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .retrieval_verification import METHODS, verify_retrieved_claim

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "lane1_data_eval/data/processed/claimcheck_dataset.csv"
HERE = Path(__file__).resolve().parent
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=[*METHODS, "all"], default="bm25")
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    if args.top_k < 1 or (args.limit is not None and args.limit < 1):
        ap.error("--top-k and --limit must be positive")
    df = pd.read_csv(DATA, index_col=0)
    df = df[(df.split == args.split) & (df.source == "fever_symmetric")]
    if args.limit is not None:
        df = df.head(args.limit)
    if df.empty:
        raise ValueError("No test rows found")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    methods = METHODS if args.method == "all" else (args.method,)
    summaries = []
    for method in methods:
        output = []
        print(f"Evaluating {method} on {args.split}: {len(df)} claims", flush=True)
        for number, (idx, row) in enumerate(df.iterrows(), 1):
            result = verify_retrieved_claim(str(row.claim), method=method,
                                            k=args.top_k, strategy="conflict_abstain")
            evidence = result["evidence"]
            conservative = result["verdict"]
            # Top-1 uses the same already computed evidence-level predictions.
            from .claim_verdict import aggregate_predictions
            top1 = aggregate_predictions(evidence, strategy="top1")
            output.append({"claim_id": idx, "gold_label": row.label,
                           "retrieved_ids": " ".join(str(e["evidence_id"]) for e in evidence),
                           "top1_label": top1["label"], "top1_confidence": top1["confidence"],
                           "conflict_abstain_label": conservative["label"],
                           "conflict_abstain_confidence": conservative["confidence"],
                           "conflict": bool(conservative["counts"]["SUPPORTS"] and
                                            conservative["counts"]["REFUTES"])})
            if number % 25 == 0:
                print(f"{method}: processed {number}/{len(df)} claims", flush=True)
        results = pd.DataFrame(output)
        stem = f"aggregation_{method}_{args.split}_top{args.top_k}"
        results.to_csv(out_dir / f"{stem}_predictions.csv", index=False)
        rows = []
        for strategy in ("top1", "conflict_abstain"):
            pred = results[f"{strategy}_label"]
            rows.append({"strategy": strategy, "method": method, "split": args.split,
                         "n": len(results), "top_k": args.top_k,
                         "accuracy": accuracy_score(results.gold_label, pred),
                         "macro_f1_3class": f1_score(results.gold_label, pred, labels=LABELS, average="macro", zero_division=0),
                         "macro_f1_supports_refutes": f1_score(results.gold_label, pred, labels=LABELS[:2], average="macro", zero_division=0),
                         "nei_predictions": int((pred == LABELS[2]).sum())})
            print(f"{method} {strategy} confusion matrix:\n{confusion_matrix(results.gold_label, pred, labels=LABELS)}")
        summary = pd.DataFrame(rows)
        summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
        print(summary.to_string(index=False))
        summaries.extend(rows)
    pd.DataFrame(summaries).to_csv(out_dir / f"aggregation_comparison_{args.split}_top{args.top_k}.csv", index=False)
    if args.limit is not None:
        print("WARNING: this is a limited diagnostic sample, not the full split result")

if __name__ == "__main__":
    main()
