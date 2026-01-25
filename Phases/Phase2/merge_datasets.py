# =============================================================
# Merging enron and CERT dataset
# =============================================================
import pandas as pd

CERT_PATH = "./data/cert_processed/cert_incidents.csv"
ENRON_PATH = "./data/enron_processed/enron_incidents.csv"
OUTPUT_PATH = "./data/real_incidents.csv"

def merge():
    cert = pd.read_csv(CERT_PATH)
    enron = pd.read_csv(ENRON_PATH)

    # Normalize labels (safety)
    label_map = {
        "Insider Misuse": "Insider Misuse",
        "Data Breach": "Data Breach",
        "Phishing": "Phishing"
    }

    cert["incident_type"] = cert["incident_type"].map(label_map)
    enron["incident_type"] = enron["incident_type"].map(label_map)

    df = pd.concat([cert, enron], ignore_index=True)

    # Drop empties / duplicates
    df = df.dropna(subset=["text", "incident_type"])
    df = df.drop_duplicates(subset=["text"])

    df.to_csv(OUTPUT_PATH, index=False)

    print("Merged dataset summary:")
    print(df["incident_type"].value_counts())
    print(f"\nTotal samples: {len(df)}")

if __name__ == "__main__":
    merge()
