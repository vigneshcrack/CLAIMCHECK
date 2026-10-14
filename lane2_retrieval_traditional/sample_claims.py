"""
sample_claims.py — Lane 2 sanity check: top-5 BM25 and TF-IDF results for 5 sample claims.

Picks 5 claims from the TEST split with a fixed seed (2 SUPPORTS, 2 REFUTES, 1 NOT ENOUGH INFO),
unless claim_ids are given. Lane 3 should run the SAME claim_ids (results/sample_claim_ids.txt).

    python sample_claims.py
    python sample_claims.py --ids test_123 test_456 ...
"""
import argparse
from pathlib import Path

import pandas as pd

from retrieval import CORPUS_CSV, retrieve_with_scores

HERE = Path(__file__).resolve().parent
DATASET_CSV = CORPUS_CSV.parent / "claimcheck_dataset.csv"
OUT = HERE / "results"


def pick(df: pd.DataFrame, ids, seed=7):
    if ids:
        return df[df["claim_id"].isin(ids)].set_index("claim_id").loc[ids].reset_index()
    test = df[df["split"] == "test"]
    parts = [test[test["label"] == lbl].sample(n, random_state=seed)
             for lbl, n in (("SUPPORTS", 2), ("REFUTES", 2), ("NOT ENOUGH INFO", 1))]
    return pd.concat(parts).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    df = pd.read_csv(DATASET_CSV)
    sample = pick(df, a.ids)
    OUT.mkdir(exist_ok=True)
    (OUT / "sample_claim_ids.txt").write_text("\n".join(sample["claim_id"]) + "\n")

    rows, md = [], ["# Lane 2 — sanity check on 5 sample claims (top-5)\n"]
    for _, c in sample.iterrows():
        gold = c["gold_evidence_id"] if isinstance(c["gold_evidence_id"], str) else None
        md += [f"## {c['claim_id']} — gold label: {c['label']}", "", f"> {c['claim']}", "",
               f"Gold evidence: `{gold}`" if gold else "Gold evidence: none (NOT ENOUGH INFO)", ""]
        for m in ("bm25", "tfidf"):
            hits = retrieve_with_scores(c["claim"], a.k, method=m)
            rank = next((i + 1 for i, (e, _, _) in enumerate(hits) if e == gold), None)
            md += [f"**{m.upper()}** — gold in top-{a.k}: " + (f"yes (rank {rank})" if rank else "no"), "",
                   "| rank | score | evidence_id | gold? | text |", "|---|---|---|---|---|"]
            for r, (eid, s, txt) in enumerate(hits, 1):
                md.append(f"| {r} | {s:.3f} | {eid} | {'✓' if eid == gold else ''} | {txt[:150].replace('|', '/')} |")
                rows.append(dict(claim_id=c["claim_id"], claim=c["claim"], label=c["label"], retriever=m,
                                 rank=r, evidence_id=eid, score=round(s, 4), is_gold=eid == gold, text=txt))
            md.append("")
    pd.DataFrame(rows).to_csv(OUT / "sample_5_claims.csv", index=False)
    (OUT / "sample_5_claims.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
