import pandas as pd
from nli_model import verify_batch, clean_fever_text

df = pd.read_csv("../lane1_data_eval/data/processed/claimcheck_dataset.csv", index_col=0)
test = df[(df.split == "test") & (df.source == "fever_symmetric")]

preds = verify_batch(test.claim.tolist(), test.evidence.fillna("").tolist())

out = pd.DataFrame({
    "claim_id": test.index,
    "predicted_label": [p["label"] for p in preds],
    "confidence": [p["confidence"] for p in preds],
    "gold_label": test.label.values,
})
out.to_csv("results/predictions_nli.csv", index=False)
print((out.predicted_label == out.gold_label).mean())