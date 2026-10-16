import pandas as pd
from nli_model import verify_batch, clean_fever_text

df = pd.read_csv("../lane1_data_eval/data/processed/claimcheck_dataset.csv", index_col=0)
test = df[(df.split == "test") & (df.source == "fever_symmetric")]
out = pd.read_csv("results/predictions_nli.csv").merge(
    test[["claim", "evidence"]], left_on="claim_id", right_index=True)
nei = out[out.predicted_label == "NOT ENOUGH INFO"]

print(nei.confidence.describe())
print(nei.gold_label.value_counts())
for _, r in nei.sample(10, random_state=0).iterrows():
    print(r.gold_label, "|", r.claim, "|", r.evidence[:200], "\n")