import pandas as pd
df = pd.read_csv("../lane1_data_eval/data/processed/claimcheck_dataset.csv", index_col=0)

print(df.shape)
print(df.columns.tolist())
print(df.source.value_counts())
print(df.groupby(["source", "split"]).label.value_counts())
print(df.evidence.fillna("").str.strip().eq("").sum(), "empty evidence rows")