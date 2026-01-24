import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from core.constants import PHASE_ORDER
import json

PHASE_WEIGHTS = {
    "Identification": 1.0,
    "Containment": 0.8,
    "Eradication": 0.6,
    "Recovery": 0.5,
    "Post-Incident": 0.4,
}


DATA_PATH = "./data/real_incidents_balanced.csv"
VECTORIZER_PATH = "./models/tfidf.pkl"

with open("knowledge/action_kb.json") as f:
    ACTION_KB = json.load(f)

vectorizer = joblib.load(VECTORIZER_PATH)
df = pd.read_csv(DATA_PATH)
X_hist = vectorizer.transform(df["text"])

ACTION_MAP = {
    "Phishing": ["IR-ID-01", "IR-CON-02", "IR-POST-01"],
    "Insider Misuse": ["IR-ID-01", "IR-CON-02", "IR-POST-01"],
    "Data Breach": ["IR-ID-01", "IR-CON-01", "IR-POST-01"],
}


def recommend_actions(text: str, incident_type: str, cls_conf: float, top_k: int = 5):

    x_new = vectorizer.transform([text])
    similarities = cosine_similarity(x_new, X_hist)[0]

    top_idx = similarities.argsort()[-top_k:][::-1]
    total_sim = sum(similarities[i] for i in top_idx)

    action_scores = {}
    similar_incidents = []

    for idx in top_idx:
        sim = similarities[idx]
        label = df.iloc[idx]["incident_type"]
        snippet = df.iloc[idx]["text"][:500]

        similar_incidents.append(
            {"incident_type": label, "similarity": float(sim), "text": snippet}
        )

        for action_id in ACTION_MAP.get(label, []):
            phase = ACTION_KB[action_id]["phase"]
            phase_weight = PHASE_WEIGHTS.get(phase, 0.3)

            weighted_score = sim * phase_weight * cls_conf
            action_scores[action_id] = action_scores.get(action_id, 0) + weighted_score

        ranked_actions = []
        max_score = max(action_scores.values()) if action_scores else 1.0

        for action_id, score in action_scores.items():
            phase = ACTION_KB[action_id]["phase"]
            phase_rank = PHASE_ORDER.get(phase, 99)

            ranked_actions.append(
                {
                    "action_id": action_id,
                    "confidence": round((score / max_score) * 100, 2),
                    "phase": phase,
                    "phase_rank": phase_rank,
                }
            )

        ranked_actions.sort(key=lambda x: (x["phase_rank"], -x["confidence"]))

    return ranked_actions, similar_incidents
