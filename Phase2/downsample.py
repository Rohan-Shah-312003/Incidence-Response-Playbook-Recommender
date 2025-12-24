# ========================================================================
# For downsampling the real dataset to 
# prevent overtraining and overfitting of model
# ========================================================================

import pandas as pd

INPUT = "./data/real_incidents.csv"
OUTPUT = "./data/real_incidents_balanced.csv"

TARGET_PER_CLASS = 3000

df = pd.read_csv(INPUT)

dfs = []
for label in df["incident_type"].unique():
    subset = df[df["incident_type"] == label]

    if len(subset) > TARGET_PER_CLASS:
        subset = subset.sample(TARGET_PER_CLASS, random_state=42)

    dfs.append(subset)

balanced = pd.concat(dfs).sample(frac=1, random_state=42)

balanced.to_csv(OUTPUT, index=False)

print(balanced["incident_type"].value_counts())
print(f"Total samples: {len(balanced)}")
