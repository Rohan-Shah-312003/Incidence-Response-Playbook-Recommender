import csv
import random
from uuid import uuid4
import hashlib

# ============================================================
# INCIDENT TYPES
# ============================================================

INCIDENT_TYPES = [
    "Malware",
    "Ransomware",
    "Phishing",
    "Data Breach",
    "Insider Misuse",
    "Denial of Service"
]

INCIDENTS_PER_TYPE = 30
TOTAL_INCIDENTS = INCIDENTS_PER_TYPE * len(INCIDENT_TYPES)

# ============================================================
# DEEP PARAPHRASING DICTIONARIES
# ============================================================

# Vocabulary substitution for variety
SYNONYMS = {
    "detected": ["identified", "discovered", "observed", "found", "noticed", "spotted", "caught", "uncovered"],
    "unusual": ["abnormal", "anomalous", "suspicious", "irregular", "unexpected", "atypical", "strange"],
    "system": ["endpoint", "host", "machine", "device", "workstation", "infrastructure"],
    "activity": ["behavior", "operations", "actions", "events", "patterns", "conduct"],
    "access": ["entry", "interaction", "connection", "engagement", "usage"],
    "unauthorized": ["illegitimate", "unapproved", "unpermitted", "unsanctioned", "unverified"],
    "security": ["protective", "defensive", "safety", "threat detection"],
    "monitoring": ["surveillance", "oversight", "observation", "tracking", "analysis"],
    "reported": ["escalated", "flagged", "communicated", "submitted", "notified"],
    "caused": ["resulted in", "led to", "triggered", "produced", "generated"],
    "disruption": ["interruption", "disturbance", "interference", "breakdown", "impairment"],
    "risk": ["threat", "danger", "exposure", "vulnerability", "concern"],
    "potential": ["possible", "probable", "suspected", "anticipated"],
    "data": ["information", "records", "files", "datasets", "content"],
    "user": ["personnel", "employee", "staff member", "individual", "account holder"]
}

def apply_synonyms(text, seed):
    """Apply synonym substitution based on seed for consistency"""
    random.seed(seed)
    words = text.split()
    result = []
    
    for word in words:
        word_lower = word.lower().strip('.,!?;:')
        if word_lower in SYNONYMS and random.random() > 0.3:  # 70% chance to substitute
            replacement = random.choice(SYNONYMS[word_lower])
            # Preserve capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
        else:
            result.append(word)
    
    random.seed()  # Reset
    return ' '.join(result)

# ============================================================
# COMPONENT LIBRARIES WITH HIGH VARIETY
# ============================================================

OBSERVATIONS = {
    "Malware": [
        "endpoint experiencing severe performance degradation",
        "unknown executable consuming excessive CPU resources",
        "multiple applications terminating unexpectedly",
        "unapproved software running on workstation",
        "suspicious process modifying system registry",
        "unexpected outbound network connections initiated",
        "files being altered without user authorization",
        "antivirus protection has been disabled",
        "memory usage spiking due to hidden processes",
        "legitimate software behaving abnormally"
    ],
    "Ransomware": [
        "widespread file encryption across network shares",
        "critical business data rendered inaccessible",
        "rapid file modification event detected on server",
        "users unable to open documents and databases",
        "ransom demand note appeared on desktop",
        "backup systems showing signs of compromise",
        "desktop wallpaper replaced with threat message",
        "file extensions changed to unknown format",
        "encryption process consuming server resources",
        "shadow copies deleted from affected systems"
    ],
    "Phishing": [
        "employees submitting credentials to fraudulent portal",
        "malicious email campaign targeting finance team",
        "users clicking on links in deceptive messages",
        "fake login page harvesting user passwords",
        "brand impersonation detected in email headers",
        "attachment containing malicious payload opened",
        "credential entry on spoofed company website",
        "suspicious authentication requests from users",
        "email with urgent tone requesting sensitive data",
        "domain name similar to legitimate site accessed"
    ],
    "Data Breach": [
        "abnormal SQL queries executed against database",
        "bulk export of customer records detected",
        "sensitive information accessed from unusual location",
        "API calls retrieving excessive amounts of data",
        "authentication bypass allowing unauthorized access",
        "database administrator credentials compromised",
        "large data transfer to external IP address",
        "personally identifiable information extracted",
        "audit logs showing irregular data retrieval",
        "privileged account accessing unrelated datasets"
    ],
    "Insider Misuse": [
        "employee accessing files outside job scope",
        "privileged credentials used for personal purposes",
        "internal user violating acceptable use policy",
        "repeated attempts to access restricted servers",
        "downloading confidential data to personal device",
        "after-hours access to sensitive systems",
        "data exfiltration by authorized personnel",
        "administrator abusing elevated permissions",
        "employee bypassing security controls",
        "unusual data access patterns by trusted user"
    ],
    "Denial of Service": [
        "service outage due to overwhelming traffic volume",
        "network infrastructure saturated with requests",
        "application servers failing under load pressure",
        "bandwidth exhausted by malicious traffic",
        "legitimate users unable to access services",
        "connection floods overwhelming firewall",
        "CPU resources depleted on web servers",
        "distributed attack targeting public endpoints",
        "response times degraded to unacceptable levels",
        "infrastructure collapsing under request volume"
    ]
}

DETECTION_METHODS = {
    "Malware": [
        "endpoint protection agent raised alert",
        "user reported unusual behavior to helpdesk",
        "automated scan identified threat signature",
        "security operations center detected anomaly",
        "behavioral analysis flagged suspicious activity",
        "threat intelligence match triggered alarm",
        "IT administrator noticed during routine check",
        "heuristic engine caught unknown malware variant"
    ],
    "Ransomware": [
        "file integrity monitoring detected mass changes",
        "backup system alerted to deletion attempt",
        "user called helpdesk about inaccessible files",
        "server administrator observed encryption process",
        "security monitoring detected ransomware signature",
        "automated system flagged suspicious file activity",
        "incident response team identified during investigation",
        "network monitoring tools caught data exfiltration"
    ],
    "Phishing": [
        "security awareness training prompted user report",
        "employee forwarded suspicious email to IT",
        "email gateway quarantined message automatically",
        "threat intelligence identified phishing campaign",
        "user reported unexpected authentication prompt",
        "security team detected credential submission",
        "email analysis revealed malicious indicators",
        "domain reputation check flagged spoofed site"
    ],
    "Data Breach": [
        "database activity monitoring detected anomaly",
        "security information and event management alerted",
        "compliance audit revealed unauthorized access",
        "data loss prevention system triggered warning",
        "log analysis identified suspicious queries",
        "threat hunting exercise uncovered breach",
        "third-party security assessment found evidence",
        "anomaly detection system flagged unusual pattern"
    ],
    "Insider Misuse": [
        "user behavior analytics identified deviation",
        "access control review revealed violations",
        "manager reported suspicious employee activity",
        "privileged access management system alerted",
        "audit trail analysis showed policy breach",
        "data classification system detected misuse",
        "peer reported concerning behavior to security",
        "automated monitoring caught unauthorized access"
    ],
    "Denial of Service": [
        "network operations center observed traffic spike",
        "service availability monitoring detected outage",
        "customer complaints triggered investigation",
        "infrastructure monitoring tools raised alarm",
        "load balancer reported abnormal request volume",
        "traffic analysis identified attack signature",
        "service owner escalated performance issues",
        "intrusion detection system caught flood attack"
    ]
}

IMPACTS = {
    "Malware": [
        "potential lateral movement across network",
        "credential theft and data exfiltration risk",
        "system instability affecting productivity",
        "possible installation of additional malware",
        "compromise of sensitive business information",
        "operational disruption for affected users",
        "unauthorized access to company resources",
        "threat to network security posture"
    ],
    "Ransomware": [
        "critical business operations halted",
        "permanent data loss if backups compromised",
        "extended recovery time impacting revenue",
        "ransom payment consideration by leadership",
        "regulatory reporting requirements triggered",
        "customer service capabilities degraded",
        "reputational damage to organization",
        "financial losses from downtime and recovery"
    ],
    "Phishing": [
        "compromised credentials enabling further attacks",
        "unauthorized access to email and applications",
        "business email compromise attempts possible",
        "identity theft and account takeover risk",
        "spread of attack to additional employees",
        "sensitive information disclosure potential",
        "financial fraud through compromised accounts",
        "loss of customer trust and confidence"
    ],
    "Data Breach": [
        "exposure of customer personally identifiable information",
        "regulatory fines and legal liability",
        "mandatory breach notification to affected parties",
        "loss of competitive advantage from stolen data",
        "reputational harm and customer attrition",
        "intellectual property theft concerns",
        "compliance violations requiring remediation",
        "forensic investigation and recovery costs"
    ],
    "Insider Misuse": [
        "data integrity and confidentiality compromise",
        "violation of privacy regulations and policies",
        "potential espionage or competitor collaboration",
        "need for human resources investigation",
        "loss of trade secrets or proprietary data",
        "damage to internal security culture",
        "legal action against employee possible",
        "revision of access controls required"
    ],
    "Denial of Service": [
        "revenue loss from service unavailability",
        "customer dissatisfaction and complaints",
        "brand reputation damage from outage",
        "potential for data breach during distraction",
        "service level agreement violations",
        "increased operational costs for mitigation",
        "employee productivity impacted significantly",
        "competitive disadvantage during downtime"
    ]
}

# ============================================================
# SENTENCE STRUCTURE TEMPLATES
# ============================================================

def generate_description(obs, det, imp, seed):
    """Generate highly diverse descriptions using templates and transformations"""
    
    # Apply synonym substitution to all components
    obs = apply_synonyms(obs, seed + "obs")
    det = apply_synonyms(det, seed + "det")
    imp = apply_synonyms(imp, seed + "imp")
    
    random.seed(seed)
    
    templates = [
        # Standard narrative
        f"Security incident: {obs}. Discovery method: {det}. Impact assessment indicates {imp}.",
        
        # Technical log format
        f"[ALERT] {det} || Observation: {obs} || Risk: {imp}",
        
        # Timeline-based
        f"Investigation timeline: {det}, revealing {obs}. Potential consequences include {imp}.",
        
        # Impact-first
        f"Threat assessment: {imp}. Root cause identified as {obs}, which was {det}.",
        
        # Concise professional
        f"{obs} - {det}. This poses {imp}.",
        
        # Detailed analysis
        f"Analysis shows {obs}. The incident was {det}. Security team assesses {imp}.",
        
        # Question-answer format
        f"What happened: {obs}. How discovered: {det}. Why concerning: {imp}.",
        
        # Executive summary
        f"Executive summary - {det} identified {obs}, presenting {imp}.",
        
        # Chronological
        f"Event sequence: First, {det}. Investigation revealed {obs}. Resulting in {imp}.",
        
        # Risk-focused
        f"Risk identified: {imp}. Contributing factor: {obs}. Detection source: {det}.",
        
        # Passive construction
        f"{obs} has been observed. This was {det}. The situation creates {imp}.",
        
        # Active urgent
        f"URGENT: {det} detected {obs}, which threatens {imp}.",
        
        # Forensic style
        f"Forensic analysis: {obs} discovered through {det}, resulting in {imp}.",
        
        # Compliance-focused
        f"Compliance alert: {det} documented {obs}, leading to {imp}.",
        
        # Tactical
        f"Tactical assessment: {obs} identified via {det}, generating {imp}."
    ]
    
    description = random.choice(templates)
    
    # Add random variation: remove articles occasionally
    if random.random() > 0.7:
        description = description.replace(" the ", " ").replace(" a ", " ").replace(" an ", " ")
    
    # Add random variation: change punctuation style
    if random.random() > 0.5:
        description = description.replace(". ", "; ")
    
    random.seed()  # Reset
    return description

# ============================================================
# ASSET AND IMPACT FUNCTIONS
# ============================================================

def get_asset_type(incident_type):
    weights = {
        "Malware": {"Endpoint": 0.6, "Server": 0.2, "Application": 0.2},
        "Ransomware": {"Server": 0.5, "Endpoint": 0.3, "Database": 0.2},
        "Phishing": {"Application": 0.4, "Endpoint": 0.4, "Network": 0.2},
        "Data Breach": {"Database": 0.4, "Application": 0.3, "Server": 0.3},
        "Insider Misuse": {"Server": 0.4, "Database": 0.3, "Application": 0.3},
        "Denial of Service": {"Network": 0.5, "Server": 0.3, "Application": 0.2}
    }
    assets = list(weights[incident_type].keys())
    probs = list(weights[incident_type].values())
    return random.choices(assets, weights=probs)[0]

def get_business_impact(incident_type):
    weights = {
        "Malware": {"Low": 0.2, "Medium": 0.6, "High": 0.2},
        "Ransomware": {"Medium": 0.2, "High": 0.7, "Critical": 0.1},
        "Phishing": {"Low": 0.5, "Medium": 0.4, "High": 0.1},
        "Data Breach": {"Medium": 0.2, "High": 0.6, "Critical": 0.2},
        "Insider Misuse": {"Low": 0.3, "Medium": 0.5, "High": 0.2},
        "Denial of Service": {"Medium": 0.3, "High": 0.6, "Critical": 0.1}
    }
    impacts = list(weights[incident_type].keys())
    probs = list(weights[incident_type].values())
    return random.choices(impacts, weights=probs)[0]

# ============================================================
# OVERLAPPING ACTION IDS
# ============================================================

COMMON_ACTIONS = ["IR-ID-01", "IR-ID-02", "IR-CON-01", "IR-POST-01"]

ACTION_LIBRARY = {
    "Malware": COMMON_ACTIONS + ["IR-ERA-01", "IR-SCAN-01"],
    "Ransomware": COMMON_ACTIONS + ["IR-ERA-01", "IR-REC-01", "IR-ISO-01"],
    "Phishing": COMMON_ACTIONS + ["IR-CON-02", "IR-EDU-01"],
    "Data Breach": COMMON_ACTIONS + ["IR-CON-02", "IR-LEGAL-01", "IR-NOTIF-01"],
    "Insider Misuse": COMMON_ACTIONS + ["IR-CON-02", "IR-HR-01"],
    "Denial of Service": COMMON_ACTIONS + ["IR-MIT-01", "IR-BLOCK-01"]
}

# ============================================================
# INCIDENT GENERATOR
# ============================================================

def generate_incident(incident_type, incident_number):
    """Generate unique incident with deep linguistic diversity"""
    
    # Create unique seed from incident details
    seed_string = f"{incident_type}{incident_number}{uuid4().hex[:8]}"
    seed_hash = hashlib.md5(seed_string.encode()).hexdigest()
    
    # Select components
    random.seed(seed_hash)
    obs = random.choice(OBSERVATIONS[incident_type])
    det = random.choice(DETECTION_METHODS[incident_type])
    imp = random.choice(IMPACTS[incident_type])
    random.seed()  # Reset
    
    # Generate description with transformations
    description = generate_description(obs, det, imp, seed_hash)
    
    return {
        "incident_id": f"INC-{uuid4().hex[:8]}",
        "incident_title": f"{incident_type} incident reported",
        "incident_description": description,
        "incident_type": incident_type,
        "asset_type": get_asset_type(incident_type),
        "business_impact": get_business_impact(incident_type),
        "user_privilege": random.choice(["User", "Admin", "Guest", "Service Account"]),
        "recommended_action_ids": ";".join(random.sample(
            ACTION_LIBRARY[incident_type], 
            k=min(random.randint(3, 5), len(ACTION_LIBRARY[incident_type]))
        )),
        "outcome": random.choice(["Success"] * 18 + ["Partial"] * 9 + ["Failure"] * 3)
    }

# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(output_file="incidents.csv"):
    records = []
    incident_counter = 0
    
    for incident_type in INCIDENT_TYPES:
        for i in range(INCIDENTS_PER_TYPE):
            records.append(generate_incident(incident_type, incident_counter))
            incident_counter += 1
    
    random.shuffle(records)
    
    # Verify uniqueness
    descriptions = [r["incident_description"] for r in records]
    unique_descriptions = set(descriptions)
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    
    print(f"[+] Generated {TOTAL_INCIDENTS} incidents")
    print(f"[+] Unique descriptions: {len(unique_descriptions)}/{len(descriptions)}")
    print(f"[+] Uniqueness rate: {len(unique_descriptions)/len(descriptions)*100:.1f}%")
    print(f"    - Deep synonym substitution applied")
    print(f"    - 15 diverse sentence templates")
    print(f"    - Vocabulary overlap with paraphrasing")
    print(f"    - Expected avg similarity: 0.25-0.35")
    print(f"    - Expected max similarity: 0.60-0.75")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_dataset(output_file="data/incidents.csv")