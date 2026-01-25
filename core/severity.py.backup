def compute_severity(
    incident_type: str,
    confidence: float,
    actions: list
) -> dict:
    """
    Compute severity level based on context and response urgency.
    """

    severity_score = 0

    # Base score by incident type
    base_scores = {
        "Insider Misuse": 3,
        "Data Breach": 4,
        "Phishing": 2,
        "Malware": 4,
        "Ransomware": 5
    }

    severity_score += base_scores.get(incident_type, 2)

    # Confidence influence
    if confidence > 0.7:
        severity_score += 1
    elif confidence < 0.4:
        severity_score -= 1

    # Presence of containment actions increases urgency
    for a in actions:
        if a["phase"] == "Containment":
            severity_score += 1
            break

    # Clamp score
    severity_score = max(1, min(severity_score, 5))

    severity_map = {
        1: "Low",
        2: "Low",
        3: "Medium",
        4: "High",
        5: "Critical"
    }

    return {
        "level": severity_map[severity_score],
        "score": severity_score
    }
