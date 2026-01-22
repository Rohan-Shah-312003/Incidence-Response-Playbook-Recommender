from core.classifier import classify_incident
from core.similarity import recommend_actions
from core.explainer import explain_actions
from core.situation import generate_situation_assessment
from core.severity import compute_severity


def run_pipeline(incident_text: str):
    result = {}

    label, cls_conf = classify_incident(incident_text)
    result["incident_type"] = label
    result["classification_confidence"] = cls_conf

    actions, similar_incidents = recommend_actions(incident_text)
    result["actions"] = actions
    result["similar_incidents"] = similar_incidents
    severity = compute_severity(label, cls_conf, actions)
    result["severity"] = severity

    situation = generate_situation_assessment(
        incident_text,
        label,
        cls_conf,
        similar_incidents
    )

    result["situation_assessment"] = situation


    explanations = explain_actions(incident_text, actions)
    result["explanations"] = explanations

    print("ACTIONS:", actions)
    print("EXPLANATIONS:", explanations)


    return result
