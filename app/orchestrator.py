from core.classifier import classify_incident
from core.similarity import recommend_actions
from core.explainer import explain_actions

def run_pipeline(incident_text: str):
    result = {}

    label, cls_conf = classify_incident(incident_text)
    result["incident_type"] = label
    result["classification_confidence"] = cls_conf

    actions = recommend_actions(incident_text)
    result["actions"] = actions

    explanations = explain_actions(incident_text, actions)
    result["explanations"] = explanations

    return result
