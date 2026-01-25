import pandas as pd

df = pd.read_csv("data/incidents.csv")

print("\n=== INCIDENT TYPE DISTRIBUTION ===")
print(df["incident_type"].value_counts())

print("\n=== OUTCOME DISTRIBUTION ===")
print(df["outcome"].value_counts())

print("\n=== BUSINESS IMPACT DISTRIBUTION ===")
print(df["business_impact"].value_counts())

print("\n=== SAMPLE RECORD ===")
print(df.sample(1).T)


# use uv for package management
# run uv run dataset_stats.py to get the stats