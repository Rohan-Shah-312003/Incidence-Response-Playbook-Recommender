"""
Enhanced similarity recommender using sentence embeddings
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load action knowledge base
KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "action_kb.json"
with open(KB_PATH) as f:
    ACTION_KB = json.load(f)

# Phase ordering for action prioritization
PHASE_ORDER = {
    "Identification": 1,
    "Containment": 2,
    "Eradication": 3,
    "Recovery": 4,
    "Post-Incident": 5
}

PHASE_WEIGHTS = {
    "Identification": 1.0,
    "Containment": 0.9,
    "Eradication": 0.7,
    "Recovery": 0.6,
    "Post-Incident": 0.4,
}

# Action mapping by incident type
ACTION_MAP = {
    "Phishing": ["IR-ID-01", "IR-CON-02", "IR-CON-03", "IR-ERAD-01", "IR-POST-01", "IR-POST-02"],
    "Insider Misuse": ["IR-ID-01", "IR-ID-02", "IR-CON-02", "IR-ERAD-02", "IR-POST-01", "IR-POST-02"],
    "Data Breach": ["IR-ID-01", "IR-ID-02", "IR-CON-01", "IR-ERAD-01", "IR-ERAD-02", "IR-POST-01"],
    "Malware": ["IR-ID-01", "IR-ID-02", "IR-CON-01", "IR-ERAD-01", "IR-REC-01", "IR-POST-01"],
    "Ransomware": ["IR-ID-01", "IR-CON-01", "IR-ERAD-01", "IR-REC-01", "IR-REC-02", "IR-POST-01"],
    "Denial of Service": ["IR-ID-01", "IR-CON-01", "IR-REC-01", "IR-REC-02", "IR-POST-01"]
}

# Try to load enhanced recommender
RECOMMENDER_PATH = Path("models/enhanced_similarity")

if RECOMMENDER_PATH.exists():
    print("Loading enhanced similarity recommender...")
    from enhanced_similarity import EnhancedSimilarityRecommender
    
    _recommender = EnhancedSimilarityRecommender.load(RECOMMENDER_PATH)
    _use_enhanced_similarity = True
else:
    print("Enhanced recommender not found, using legacy approach...")
    # Legacy approach
    DATA_PATH = "./data/real_incidents_balanced.csv"
    VECTORIZER_PATH = "./models/tfidf.pkl"
    
    vectorizer = joblib.load(VECTORIZER_PATH)
    df = pd.read_csv(DATA_PATH)
    X_hist = vectorizer.transform(df["text"])
    _use_enhanced_similarity = False


def recommend_actions(
    text: str, 
    incident_type: str, 
    cls_conf: float, 
    top_k: int = 5
):
    """
    Recommend response actions based on incident similarity
    
    Args:
        text: Incident description
        incident_type: Predicted incident type
        cls_conf: Classification confidence
        top_k: Number of similar incidents to retrieve
        
    Returns:
        (recommended_actions, similar_incidents)
    """
    if _use_enhanced_similarity:
        # Use enhanced recommender
        return _recommender.recommend(
            incident_text=text,
            incident_type=incident_type,
            classification_confidence=cls_conf,
            top_k=top_k
        )
    else:
        # Legacy approach
        return _legacy_recommend(text, incident_type, cls_conf, top_k)


def _legacy_recommend(text: str, incident_type: str, cls_conf: float, top_k: int = 5):
    """Legacy recommendation approach (TF-IDF based)"""
    x_new = vectorizer.transform([text])
    similarities = cosine_similarity(x_new, X_hist)[0]

    top_idx = similarities.argsort()[-top_k:][::-1]

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
            if action_id not in ACTION_KB:
                continue
                
            phase = ACTION_KB[action_id]["phase"]
            phase_weight = PHASE_WEIGHTS.get(phase, 0.5)

            weighted_score = sim * phase_weight * cls_conf
            action_scores[action_id] = action_scores.get(action_id, 0) + weighted_score

    # Rank actions
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