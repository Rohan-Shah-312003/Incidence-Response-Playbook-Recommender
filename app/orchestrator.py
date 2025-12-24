from core.classifier import classify_incident
from core.similarity import recommend_actions
from core.explainer import explain_actions

def run_pipeline(incident_text):
    result = {}

    # Phase 2
    label, confidence = classify_incident(incident_text)
    result["incident_type"] = label
    result["classification_confidence"] = confidence

    # Phase 3
    actions = recommend_actions(incident_text, label)
    result["recommended_actions"] = actions

    # Phase 4
    explanations = explain_actions(incident_text, actions)
    result["explanations"] = explanations

    return result
