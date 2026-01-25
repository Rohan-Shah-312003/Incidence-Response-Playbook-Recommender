"""
Enhanced severity computation with more nuanced scoring
"""

def compute_severity(
    incident_type: str,
    confidence: float,
    actions: list
) -> dict:
    """
    Compute severity level with enhanced logic
    
    Args:
        incident_type: Type of incident
        confidence: Classification confidence
        actions: Recommended actions list
        
    Returns:
        Dict with severity level and score
    """
    severity_score = 0

    # Base score by incident type (updated with more types)
    base_scores = {
        "Insider Misuse": 3,
        "Data Breach": 5,
        "Phishing": 2,
        "Malware": 4,
        "Ransomware": 5,
        "Denial of Service": 4
    }
    severity_score += base_scores.get(incident_type, 3)

    # Confidence influence (higher confidence = more certain severity)
    if confidence > 0.85:
        severity_score += 1
    elif confidence < 0.5:
        severity_score -= 1

    # Action urgency indicators
    high_priority_phases = ['Containment', 'Eradication']
    urgent_actions = sum(
        1 for a in actions 
        if a['phase'] in high_priority_phases and a['confidence'] > 70
    )
    
    if urgent_actions >= 3:
        severity_score += 1
    elif urgent_actions >= 2:
        severity_score += 0.5

    # Number of recommended actions (more actions = more complex)
    if len(actions) > 6:
        severity_score += 0.5

    # Clamp score between 1 and 5
    severity_score = max(1, min(int(round(severity_score)), 5))

    # Map to severity levels
    severity_map = {
        1: "Low",
        2: "Low-Medium",
        3: "Medium",
        4: "High",
        5: "Critical"
    }

    return {
        "level": severity_map.get(severity_score, "Medium"),
        "score": severity_score
    }
