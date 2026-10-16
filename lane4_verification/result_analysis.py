import inspect
import sys
 
import pandas as pd
 
sys.path.insert(0, "../lane1_data_eval")
from scripts import evaluation as ev
 
REAL = ["SUPPORTS", "REFUTES"]
ALL3 = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
 
print("Harness default LABELS:", getattr(ev, "LABELS", "<not found>"))
for name in ["bootstrap_ci", "paired_bootstrap_test"]:
    fn = getattr(ev, name, None)
    print(f"{name}{inspect.signature(fn) if fn else '  <-- NOT FOUND'}")
print()
 
 
def try_calls(label, *calls):
    for c in calls:
        try:
            print(label, c())
            return
        except Exception as e:
            print(f"  [{label}] attempt failed: {type(e).__name__}: {e}")
    print(f"  [{label}] all attempts failed - check the signature above")

nli = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/predictions_nli.csv")
y, p = nli.gold_label.tolist(), nli.predicted_label.tolist()
 
print("accuracy                    :", ev.accuracy(y, p))
print("macro_f1 (SUPPORTS/REFUTES) :", ev.macro_f1(y, p, labels=REAL))
print("macro_f1 (harness default)  :", ev.macro_f1(y, p), "<- includes NEI, don't headline")
print("NEI predictions on 2-class rows:", sum(1 for x in p if x == "NOT ENOUGH INFO"))
print(ev.confusion_matrix_df(y, p, labels=ALL3), "\n")
 
hits = ev.correctness(y, p)
try_calls("accuracy CI:",
          lambda: ev.bootstrap_ci(hits),
          lambda: ev.bootstrap_ci(ev.accuracy, y, p),
          lambda: ev.bootstrap_ci(y, p, ev.accuracy))
 
if len(sys.argv) > 2:
    other = pd.read_csv(sys.argv[2])
    m = nli.merge(other, on="claim_id", suffixes=("_a", "_b"))
    assert (m.gold_label_a == m.gold_label_b).all(), "gold labels disagree after join"
    yg = m.gold_label_a.tolist()
    pa, pb = m.predicted_label_a.tolist(), m.predicted_label_b.tolist()
    ha, hb = ev.correctness(yg, pa), ev.correctness(yg, pb)
    try_calls("paired test:",
              lambda: ev.paired_bootstrap_test(ha, hb),
              lambda: ev.paired_bootstrap_test(ev.accuracy, yg, pa, pb),
              lambda: ev.paired_bootstrap_test(yg, pa, pb, ev.accuracy))

