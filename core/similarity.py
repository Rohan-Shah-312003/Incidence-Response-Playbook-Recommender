"""
Enhanced similarity recommender using sentence embeddings
UPDATED: Now supports all 6 incident types with expanded action mappings
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
    "Post-Incident": 5,
}

PHASE_WEIGHTS = {
    "Identification": 1.0,
    "Containment": 0.9,
    "Eradication": 0.7,
    "Recovery": 0.6,
    "Post-Incident": 0.4,
}

# ================================================================
# EXPANDED ACTION MAPPING - ALL 6 INCIDENT TYPES
# Maps each incident type to appropriate response actions
# ================================================================

ACTION_MAP = {
    "Phishing": [
        "IR-ID-01",  # Confirm incident scope
        "IR-ID-02",  # Preserve incident evidence
        "IR-CON-02",  # Disable compromised account
        "IR-CON-03",  # Notify affected user
        "IR-ERAD-01",  # Remove malicious artifacts (phishing emails)
        "IR-ERAD-02",  # Reset credentials and access keys
        "IR-POST-01",  # Conduct post-incident review
        "IR-POST-02",  # Update security awareness and controls
        "IR-POST-03",  # Update threat intelligence
    ],
    "Insider Misuse": [
        "IR-ID-01",  # Confirm incident scope
        "IR-ID-02",  # Preserve incident evidence
        "IR-CON-01",  # Isolate affected system or account
        "IR-CON-02",  # Disable compromised account
        "IR-ERAD-02",  # Reset credentials and access keys
        "IR-REC-01",  # Validate system integrity
        "IR-POST-01",  # Conduct post-incident review
        "IR-POST-02",  # Update security awareness and controls
        "IR-POST-04",  # Compliance and regulatory reporting
    ],
    "Data Breach": [
        "IR-ID-01",  # Confirm incident scope
        "IR-ID-02",  # Preserve incident evidence
        "IR-CON-01",  # Isolate affected system or account
        "IR-CON-05",  # Block malicious infrastructure
        "IR-ERAD-01",  # Remove malicious artifacts
        "IR-ERAD-02",  # Reset credentials and access keys
        "IR-REC-01",  # Validate system integrity
        "IR-REC-04",  # Implement enhanced monitoring
        "IR-POST-01",  # Conduct post-incident review
        "IR-POST-03",  # Update threat intelligence
        "IR-POST-04",  # Compliance and regulatory reporting
    ],
    "Malware": [
        "IR-ID-01",  # Confirm incident scope
        "IR-ID-02",  # Preserve incident evidence
        "IR-ID-03",  # Identify malware variant and indicators
        "IR-CON-01",  # Isolate affected system or account
        "IR-CON-04",  # Implement network segmentation
        "IR-CON-05",  # Block malicious infrastructure (C2 servers)
        "IR-ERAD-01",  # Remove malicious artifacts (malware files)
        "IR-ERAD-03",  # Patch vulnerabilities
        "IR-ERAD-04",  # Remove persistence mechanisms
        "IR-REC-01",  # Validate system integrity
        "IR-REC-04",  # Implement enhanced monitoring
        "IR-POST-01",  # Conduct post-incident review
        "IR-POST-03",  # Update threat intelligence
    ],
    "Ransomware": [
        "IR-ID-01",  # Confirm incident scope
        "IR-ID-02",  # Preserve incident evidence
        "IR-CON-01",  # Isolate affected system (prevent spread)
        "IR-CON-04",  # Implement network segmentation
        "IR-CON-05",  # Block malicious infrastructure
        "IR-CON-06",  # Disable vulnerable services
        "IR-ERAD-01",  # Remove malicious artifacts
        "IR-ERAD-04",  # Remove persistence mechanisms
        "IR-REC-01",  # Validate system integrity
        "IR-REC-03",  # Restore from clean backups
        "IR-REC-04",  # Implement enhanced monitoring
        "IR-POST-01",  # Conduct post-incident review
        "IR-POST-03",  # Update threat intelligence
        "IR-POST-04",  # Compliance and regulatory reporting
    ],
    "Denial of Service": [
        "IR-ID-01",  # Confirm incident scope
        "IR-CON-01",  # Isolate affected system (implement filtering)
        "IR-CON-04",  # Implement network segmentation
        "IR-CON-05",  # Block malicious infrastructure (attack sources)
        "IR-REC-01",  # Validate system integrity
        "IR-REC-02",  # Restore normal operations
        "IR-REC-04",  # Implement enhanced monitoring
        "IR-POST-01",  # Conduct post-incident review
        "IR-POST-02",  # Update security controls
        "IR-POST-03",  # Update threat intelligence
    ],
}

# Try to load enhanced recommender
RECOMMENDER_PATH = Path("models/enhanced_similarity")

if RECOMMENDER_PATH.exists():
    print("Loading enhanced similarity recommender...")
    try:
        from Phases.Phase5.enhanced_similarity import EnhancedSimilarityRecommender

        _recommender = EnhancedSimilarityRecommender.load(RECOMMENDER_PATH)
        _use_enhanced_similarity = True
    except ImportError:
        print("  ⚠️  Could not import EnhancedSimilarityRecommender")
        _use_enhanced_similarity = False
else:
    print("Enhanced recommender not found, using legacy approach...")
    _use_enhanced_similarity = False

# Legacy approach setup
if not _use_enhanced_similarity:
    # Try to load data and vectorizer for legacy mode
    data_paths = [
        "./data/real_incidents_expanded.csv",
        "./data/real_incidents_balanced.csv",
        "../../data/real_incidents_expanded.csv",
        "../../data/real_incidents_balanced.csv",
    ]

    VECTORIZER_PATH = "./models/tfidf.pkl"

    df = None
    vectorizer = None
    X_hist = None

    try:
        vectorizer = joblib.load(VECTORIZER_PATH)

        for data_path in data_paths:
            try:
                df = pd.read_csv(data_path)
                X_hist = vectorizer.transform(df["text"])
                print(f"  ✓ Loaded data from: {data_path}")
                break
            except FileNotFoundError:
                continue

        if df is None:
            print(f"  ⚠️  Warning: No training data found.")
            print(f"  Please run: python expand_dataset.py")
    except FileNotFoundError:
        print(f"  ⚠️  Warning: Vectorizer not found at {VECTORIZER_PATH}")


def recommend_actions(text: str, incident_type: str, cls_conf: float, top_k: int = 5):
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
            top_k=top_k,
        )
    else:
        # Legacy approach
        return _legacy_recommend(text, incident_type, cls_conf, top_k)


def _legacy_recommend(text: str, incident_type: str, cls_conf: float, top_k: int = 5):
    """Legacy recommendation approach (TF-IDF based)"""

    # Fallback if data not loaded
    if df is None or vectorizer is None or X_hist is None:
        print("  ⚠️  Using rule-based fallback (no historical data)")
        return _rule_based_fallback(incident_type), []

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
                print(f"  ⚠️  Warning: Action {action_id} not in knowledge base")
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


def _rule_based_fallback(incident_type: str):
    """
    Rule-based fallback when no historical data is available
    Returns actions based purely on incident type
    """
    actions_for_type = ACTION_MAP.get(incident_type, [])

    ranked_actions = []
    for i, action_id in enumerate(actions_for_type):
        if action_id not in ACTION_KB:
            continue

        phase = ACTION_KB[action_id]["phase"]
        phase_rank = PHASE_ORDER.get(phase, 99)

        # Assign decreasing confidence based on position
        confidence = 100.0 - (i * 5)  # 100, 95, 90, 85, ...

        ranked_actions.append(
            {
                "action_id": action_id,
                "confidence": max(50.0, confidence),  # Minimum 50%
                "phase": phase,
                "phase_rank": phase_rank,
            }
        )

    ranked_actions.sort(key=lambda x: (x["phase_rank"], -x["confidence"]))
    return ranked_actions
