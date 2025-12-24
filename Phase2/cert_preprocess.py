import pandas as pd
import re

RAW_EMAIL_PATH = "./data/cert_raw/cert_email.csv"
OUTPUT_PATH = "./data/cert_processed/cert_incidents.csv"

# TEXT CLEANING
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# WEAK LABELING LOGIC
def assign_label(text):
    if any(k in text for k in ["confidential", "exfiltrat", "leak", "steal"]):
        return "Data Breach"
    if any(k in text for k in ["password", "credential", "access", "policy"]):
        return "Insider Misuse"
    return None

# MAIN PREPROCESSOR
def preprocess_cert():
    df = pd.read_csv(RAW_EMAIL_PATH)

    text_cols = []
    for col in df.columns:
        if col.lower() in [
            "subject", "subject_line", "email_subject",
            "content", "email_body", "message", "body"
        ]:
            text_cols.append(col)

    if not text_cols:
        raise RuntimeError(
            f"No usable text columns found. Columns present: {df.columns.tolist()}"
        )

    print(f"[+] Using text columns: {text_cols}")

    # Combine detected text columns
    df["text"] = df[text_cols].astype(str).agg(" ".join, axis=1)

    df["text"] = df["text"].apply(clean_text)
    df["incident_type"] = df["text"].apply(assign_label)

    # Dropping unlabeled or trivial rows
    df = df[df["incident_type"].notnull()]
    df = df[df["text"].str.len() > 50]

    df_out = df[["text", "incident_type"]].copy()
    df_out["source"] = "CERT"

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"[+] Saved {len(df_out)} CERT incident texts to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_cert()
