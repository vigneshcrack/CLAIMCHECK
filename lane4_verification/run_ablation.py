import argparse
import os
import glob
import sys
 
import pandas as pd
 
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lane1_data_eval"))
for p in ROOT.glob("*/retrieval.py"):
    sys.path.insert(0, str(p.parent))


from nli_model import verify_batch
from retrieval import retrieve_with_scores
from scripts import evaluation as ev

DATA = ROOT / "lane1_data_eval" / "data" / "processed" / "claimcheck_dataset.csv"
SEP = " "
REAL = ["SUPPORTS", "REFUTES"]
TIERS = ("gold", "top1", "top5")

def run_tiers(df, sep=SEP, **retrieve_kwargs):
    """Return one long DataFrame: a row per claim per tier."""
    # One retrieval pass at k=5; top-1 is the first hit (list is ranked best-first).
    hits = [retrieve_with_scores(c, 5, **retrieve_kwargs) for c in df.claim]
    ids = [[h[0] for h in x] for x in hits]       # (evidence_id, score, text)
    texts = [[h[2] for h in x] for x in hits]
 
    tiers = {
        "gold": (df.evidence.fillna("").tolist(), [[g] for g in df.gold_evidence_id]),
        "top1": ([sep.join(t[:1]) for t in texts], [i[:1] for i in ids]),
        "top5": ([sep.join(t[:5]) for t in texts], [i[:5] for i in ids]),
    }
 
    frames = []
    for tier, (evidence, used_ids) in tiers.items():
        print(f"verifying tier: {tier} ({len(df)} claims)")
        preds = verify_batch(df.claim.tolist(), evidence)   # cleans text internally
        frames.append(pd.DataFrame({
            "claim_id": df.index,
            "tier": tier,
            "retrieved_ids": [" ".join(x) for x in used_ids],
            "predicted_label": [p["label"] for p in preds],
            "confidence": [p["confidence"] for p in preds],
            "gold_label": df.label.values,
        }))
    return pd.concat(frames, ignore_index=True)

def summarise(res, df):
    """Per-tier metrics using Lane 1's harness, plus paired tests between tiers."""
    gold_ids = df.gold_evidence_id.tolist()
    rows, hits_by_tier = [], {}
    for tier in TIERS:
        g = res[res.tier == tier]
        y, p = g.gold_label.tolist(), g.predicted_label.tolist()
        hits = ev.correctness(y, p)
        hits_by_tier[tier] = hits
        ci = ev.bootstrap_ci(hits)
        hit_rate = sum(gi in ids.split() for gi, ids in zip(gold_ids, g.retrieved_ids)) / len(g)
        rows.append({
            "tier": tier, "n": len(g),
            "accuracy": ci["point_estimate"],
            "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
            "macro_f1_2class": ev.macro_f1(y, p, labels=REAL),
            "nei_predictions": sum(x == "NOT ENOUGH INFO" for x in p),
            "gold_evidence_in_retrieved": hit_rate,
        })
    summary = pd.DataFrame(rows)
 
    print("\n=== Ablation summary (same claims, only evidence changes) ===")
    print(summary.round(3).to_string(index=False))
 
    print("\n=== Paired bootstrap tests (accuracy) ===")
    for a, b in (("gold", "top1"), ("top1", "top5"), ("gold", "top5")):
        print(f"{a} vs {b}:", ev.paired_bootstrap_test(hits_by_tier[a], hits_by_tier[b]))
    return summary
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--method", default=None, help="e.g. tfidf (if supported)")
    ap.add_argument("--sep", default=SEP)
    a = ap.parse_args()
 
    df = pd.read_csv(DATA, index_col=0)                  # claim_id is the index
    df = df[(df.split == a.split) & (df.source == "fever_symmetric")]
    if a.limit:
        df = df.head(a.limit)
    print(f"{a.split}: {len(df)} real rows")
 
    kwargs = {"method": a.method} if a.method else {}
    res = run_tiers(df, sep=a.sep, **kwargs)
    res.to_csv("results/predictions_nli_tiers.csv", index=False)
 
    summarise(res, df).to_csv("results/ablation_summary.csv", index=False)
    print("\nsaved predictions_nli_tiers.csv and ablation_summary.csv")
