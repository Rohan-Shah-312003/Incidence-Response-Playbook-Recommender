from core.classifier import classify_incident
from core.similarity import recommend_actions
from core.explainer import explain_actions
from core.situation import generate_situation_assessment
from core.severity import compute_severity

def run_pipeline(incident_text: str):
    print("[PIPELINE] Start")

    label, cls_conf = classify_incident(incident_text)
    print("[PIPELINE] Classified")

    actions, similar_incidents = recommend_actions(
        incident_text, label, cls_conf
    )
    print("[PIPELINE] Actions + similarity done")

    severity = compute_severity(label, cls_conf, actions)
    print("[PIPELINE] Severity done")

    situation = generate_situation_assessment(
        incident_text, label, cls_conf, similar_incidents
    )
    print("[PIPELINE] Situation assessment done")

    explanations = explain_actions(incident_text, actions)
    print("[PIPELINE] Explanations done")

    return {
        "incident_type": label,
        "classification_confidence": cls_conf,
        "actions": actions,
        "similar_incidents": similar_incidents,
        "severity": severity,
        "situation_assessment": situation,
        "explanations": explanations,
    }
