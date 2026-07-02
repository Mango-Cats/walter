import pandas as pd

df = pd.read_csv("_data/D_ultra.csv")

df_label1 = df[df["label"] == 1]

df_label1.to_csv("P.csv", index=False)

print(f"Saved {len(df_label1)} samples with label = 1.")