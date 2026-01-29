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
    "Data Breach": ["IR-ID-01", "IR-CON-01", "IR-POST-01"],
}


def recommend_actions(text: str, top_k: int = 5):
    x_new = vectorizer.transform([text])
    similarities = cosine_similarity(x_new, X_hist)[0]

    top_idx = similarities.argsort()[-top_k:][::-1]
    total_sim = sum(similarities[i] for i in top_idx)

    action_scores = {}
    # supporting_incidents = []

    total_similarity = 0.0

    for idx in top_idx:
        label = df.iloc[idx]["incident_type"]
        sim = similarities[idx]
        # text = df.iloc[idx]["text"][:120] + "..."

        # supporting_incidents.append((label, sim, text))
        total_similarity += sim

        for action in ACTION_MAP.get(label, []):
            action_scores[action] = action_scores.get(action, 0) + sim

    # 🔑 Normalize action scores
    return sorted(
        [(action, score / total_sim) for action, score in action_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    # normalized_actions = []
    # for action, score in action_scores.items():
    #     confidence = score / total_similarity if total_similarity > 0 else 0
    #     normalized_actions.append((action, confidence))

    # # Sort by confidence
    # normalized_actions.sort(key=lambda x: x[1], reverse=True)

    # return normalized_actions, supporting_incidents


# demo testing:
if __name__ == "__main__":
    print("[+] Loading historical incident dataset...")

    df = pd.read_csv("./data/real_incidents_balanced.csv")

    recommender = SimilarityRecommender(df)

    # new_incident = """
    # An employee reported receiving an urgent email requesting account verification.
    # The message asked the user to click a link and enter login credentials.
    # """
    new_incident = """
    An admin mentioned facing problems in logging in their employee-only portal.
    The admin password was only known by one of their close friends.
    """

    print("\n[+] New Incident:")
    print(new_incident.strip())

    actions, evidence = recommender.recommend(new_incident, top_k=5)

    print("\n=== Recommended Response Actions ===")
    for action, score in actions:
        print(f"{action}  |  score = {score:.3f}")

    print("\n=== Similar Past Incidents ===")
    for label, sim, text in evidence:
        print(f"[{label}] similarity={sim:.3f} | {text}")
