import pandas as pd
import joblib
# from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA_PATH = "./data/real_incidents_balanced.csv"
MODEL_OUT = "./models/classifier.pkl"
VECTORIZER_OUT = "./models/tfidf.pkl"

def train_and_save():
    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["incident_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )

    classifier = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_vec, y_train)

    # 🔐 Save artifacts
    joblib.dump(classifier, MODEL_OUT)
    joblib.dump(vectorizer, VECTORIZER_OUT)

    print("[+] Model saved to:", MODEL_OUT)
    print("[+] Vectorizer saved to:", VECTORIZER_OUT)

if __name__ == "__main__":
    train_and_save()
