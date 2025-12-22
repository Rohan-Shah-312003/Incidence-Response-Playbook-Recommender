# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# def load_data(csv_path="./data/incidents.csv"):
#     return pd.read_csv(csv_path)

# def build_preprocessor():
#     text_features = "incident_description"
#     categorical_features = ["asset_type", "business_impact", "user_privilege"]

#     text_transformer = TfidfVectorizer(
#         stop_words="english",
#         ngram_range=(1, 2),
#         min_df=2
#     )

#     categorical_transformer = OneHotEncoder(handle_unknown="ignore")

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("text", text_transformer, text_features),
#             ("cat", categorical_transformer, categorical_features),
#         ]
#     )

#     return preprocessor

# def split_data(df):
#     X = df.drop(columns=["incident_type"])
#     y = df["incident_type"]

#     return train_test_split(
#         X, y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y
#     )



import pandas as pd
import re

RAW_EMAIL_PATH = "./data/cert_raw/cert_email.csv"
OUTPUT_PATH = "./data/cert_processed/cert_incidents.csv"

# -----------------------------
# TEXT CLEANING
# -----------------------------

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# WEAK LABELING LOGIC
# -----------------------------

def assign_label(text):
    if any(k in text for k in ["confidential", "exfiltrat", "leak", "steal"]):
        return "Data Breach"
    if any(k in text for k in ["password", "credential", "access", "policy"]):
        return "Insider Misuse"
    return None

# -----------------------------
# MAIN PREPROCESSOR
# -----------------------------

def preprocess_cert():
    df = pd.read_csv(RAW_EMAIL_PATH)

    # 🔑 Detect text columns safely
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

    # Drop unlabeled or trivial rows
    df = df[df["incident_type"].notnull()]
    df = df[df["text"].str.len() > 50]

    df_out = df[["text", "incident_type"]].copy()
    df_out["source"] = "CERT"

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"[+] Saved {len(df_out)} CERT incident texts to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_cert()
