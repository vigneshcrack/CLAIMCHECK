import argparse
import sys
from pathlib import Path
 
import pandas as pd
 
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lane1_data_eval"))
from scripts import evaluation as ev
 
REAL = ["SUPPORTS", "REFUTES"]
NEI = "NOT ENOUGH INFO"
 
 
def score(name, y, p):
    hits = ev.correctness(y, p)
    ci = ev.bootstrap_ci(hits)
    y_s, p_s = pd.Series(y), pd.Series(p)
    row = {
        "model": name,
        "n": len(y),
        "accuracy": ci["point_estimate"],
        "ci_lower": ci["ci_lower"],
        "ci_upper": ci["ci_upper"],
        "macro_f1_2class": ev.macro_f1(y, p, labels=REAL),
    }
    for lbl in REAL:
        mask = y_s == lbl
        row[f"recall_{lbl}"] = float((p_s[mask] == lbl).mean()) if mask.any() else float("nan")
    row["nei_predictions"] = int((p_s == NEI).sum())
    return row, hits
 
 
def to_markdown(t):
    pct = lambda x: f"{100 * x:.1f}%"
    header = ["Model", "n", "Accuracy (95% CI)", "Macro-F1", "Recall SUPPORTS",
              "Recall REFUTES", "NEI predictions"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join([
            r.model, str(r.n),
            f"{pct(r.accuracy)} ({100 * r.ci_lower:.1f}–{100 * r.ci_upper:.1f})",
            pct(r.macro_f1_2class), pct(r.recall_SUPPORTS), pct(r.recall_REFUTES),
            str(r.nei_predictions)]) + " |")
    return "\n".join(lines)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", default=str(HERE / "results/predictions_nli.csv"))
    ap.add_argument("--tfidf", default=str(HERE / "results/predictions_tfidf.csv"))
    a = ap.parse_args()
 
    nli, tfidf = pd.read_csv(a.nli), pd.read_csv(a.tfidf)
    m = nli.merge(tfidf, on="claim_id", suffixes=("_nli", "_tfidf"))
    print(f"rows: NLI {len(nli)}, TF-IDF {len(tfidf)}, shared {len(m)}")
    if len(m) < max(len(nli), len(tfidf)):
        print("WARNING: the files cover different rows; only shared rows are scored.")
    assert (m.gold_label_nli == m.gold_label_tfidf).all(), "gold labels disagree after join"
 
    y = m.gold_label_nli.tolist()
    maj = pd.Series(y).value_counts().idxmax()
    rows, hits = [], {}
    for name, p in (
        (f"Majority baseline (always {maj})", [maj] * len(y)),
        ("TF-IDF + LogReg", m.predicted_label_tfidf.tolist()),
        ("NLI (zero-shot DeBERTa-v3)", m.predicted_label_nli.tolist()),
    ):
        row, h = score(name, y, p)
        rows.append(row)
        hits[name] = h
    table = pd.DataFrame(rows)
 
    table.to_csv(HERE / "results/model_comparison.csv", index=False)
    print("\n" + to_markdown(table))
 
    names = list(hits)
    print("\nPaired bootstrap tests (accuracy, same rows):")
    for a_, b_ in ((names[2], names[1]), (names[2], names[0]), (names[1], names[0])):
        print(f"  {a_}  vs  {b_}: {ev.paired_bootstrap_test(hits[a_], hits[b_])}")
 
    agree = (m.predicted_label_nli == m.predicted_label_tfidf).mean()
    both = ((m.predicted_label_nli == m.gold_label_nli) & (m.predicted_label_tfidf == m.gold_label_nli))
    only_nli = (m.predicted_label_nli == m.gold_label_nli) & (m.predicted_label_tfidf != m.gold_label_nli)
    only_tf = (m.predicted_label_nli != m.gold_label_nli) & (m.predicted_label_tfidf == m.gold_label_nli)
    print(f"\nModels give the same label on {100 * agree:.1f}% of rows. "
          f"Both right: {both.sum()}, only NLI right: {only_nli.sum()}, "
          f"only TF-IDF right: {only_tf.sum()}, both wrong: {len(m) - both.sum() - only_nli.sum() - only_tf.sum()}")
    print(f"\nsaved model_comparison.csv in {HERE}")
 
 
if __name__ == "__main__":
    main()
