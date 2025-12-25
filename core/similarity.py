import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "./data/real_incidents_balanced.csv"
VECTORIZER_PATH = "./models/tfidf.pkl"

vectorizer = joblib.load(VECTORIZER_PATH)
df = pd.read_csv(DATA_PATH)
X_hist = vectorizer.transform(df["text"])

ACTION_MAP = {
    "Phishing": ["IR-ID-01", "IR-CON-02", "IR-POST-01"],
    "Insider Misuse": ["IR-ID-01", "IR-CON-02", "IR-POST-01"],
    "Data Breach": ["IR-ID-01", "IR-CON-01", "IR-POST-01"]
}
def recommend_actions(text: str, top_k: int = 5):
    x_new = vectorizer.transform([text])
    similarities = cosine_similarity(x_new, X_hist)[0]

    top_idx = similarities.argsort()[-top_k:][::-1]
    total_sim = sum(similarities[i] for i in top_idx)

    action_scores = {}

    for idx in top_idx:
        label = df.iloc[idx]["incident_type"]
        sim = similarities[idx]

        for action in ACTION_MAP.get(label, []):
            action_scores[action] = action_scores.get(action, 0) + sim

    # Normalize confidence
    return sorted(
        [
            (action, score / total_sim)
            for action, score in action_scores.items()
        ],
        key=lambda x: x[1],
        reverse=True
    )
