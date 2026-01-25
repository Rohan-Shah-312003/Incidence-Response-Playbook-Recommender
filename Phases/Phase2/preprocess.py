import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

def load_data(csv_path="./data/real_incidents_balanced.csv"):
    df = pd.read_csv(csv_path)

    required_cols = {"text", "incident_type"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(
            f"Dataset must contain columns {required_cols}, "
            f"found {df.columns.tolist()}"
        )

    return df

# -------------------------------------------------
# BUILD TEXT PREPROCESSOR
# -------------------------------------------------

def build_preprocessor():
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )

# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------

def split_data(df):
    X = df["text"]
    y = df["incident_type"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
