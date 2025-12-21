import csv
import random
from uuid import uuid4
from itertools import product

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

INCIDENTS_PER_TYPE = 30   # LOCKED
TOTAL_INCIDENTS = INCIDENTS_PER_TYPE * len(INCIDENT_TYPES)

# ============================================================
# FIXED METADATA (SEMANTICALLY CORRECT)
# ============================================================

ASSET_MAP = {
    "Malware": "Endpoint",
    "Ransomware": "Server",
    "Phishing": "Application",
    "Data Breach": "Application",
    "Insider Misuse": "Server",
    "Denial of Service": "Network"
}

BUSINESS_IMPACT_MAP = {
    "Malware": "Medium",
    "Ransomware": "High",
    "Phishing": "Low",
    "Data Breach": "High",
    "Insider Misuse": "Medium",
    "Denial of Service": "High"
}

# ============================================================
# ACTION IDS (RAG-READY)
# ============================================================

ACTION_LIBRARY = {
    "Malware": [
        "IR-ID-01", "IR-ID-02", "IR-CON-01", "IR-ERA-01", "IR-POST-01"
    ],
    "Ransomware": [
        "IR-ID-01", "IR-ID-02", "IR-CON-01", "IR-ERA-01",
        "IR-REC-01", "IR-POST-01"
    ],
    "Phishing": [
        "IR-ID-01", "IR-CON-02", "IR-POST-01"
    ],
    "Data Breach": [
        "IR-ID-01", "IR-CON-02", "IR-POST-01"
    ],
    "Insider Misuse": [
        "IR-ID-01", "IR-CON-02", "IR-POST-01"
    ],
    "Denial of Service": [
        "IR-ID-01", "IR-POST-01"
    ]
}

# ============================================================
# COMPOSITIONAL TEXT COMPONENTS (HIGH DIVERSITY)
# ============================================================

OBSERVATIONS = {
    "Malware": [
        "unusual system slowdown",
        "unexpected background processes",
        "high CPU usage by unknown executables",
        "frequent application crashes",
        "unauthorized software execution"
    ],
    "Ransomware": [
        "encrypted files across multiple directories",
        "inaccessible system data",
        "unauthorized encryption activity",
        "sudden loss of file access",
        "appearance of ransom notes"
    ],
    "Phishing": [
        "users entering credentials on suspicious links",
        "employees reporting deceptive emails",
        "unexpected login prompts",
        "credential submission on fake websites",
        "suspicious email campaigns"
    ],
    "Data Breach": [
        "unauthorized database queries",
        "unexpected access to sensitive records",
        "abnormal data access patterns",
        "large volumes of data extraction",
        "access from unfamiliar locations"
    ],
    "Insider Misuse": [
        "misuse of privileged system access",
        "unauthorized internal data access",
        "policy violations by internal users",
        "access beyond job responsibilities",
        "repeated access to restricted resources"
    ],
    "Denial of Service": [
        "service unavailability due to traffic spikes",
        "network congestion from excessive requests",
        "system downtime caused by overload",
        "resource exhaustion on network devices",
        "degraded service performance"
    ]
}

DETECTION = {
    "Malware": [
        "reported by an end user",
        "detected by endpoint protection software",
        "identified during routine system monitoring",
        "flagged during security scanning"
    ],
    "Ransomware": [
        "discovered during incident escalation",
        "identified by system administrators",
        "detected after file access failures",
        "reported during service disruption"
    ],
    "Phishing": [
        "reported by employees",
        "flagged during security awareness review",
        "identified through user complaints",
        "detected during email analysis"
    ],
    "Data Breach": [
        "detected during a security audit",
        "identified through abnormal access logs",
        "discovered during compliance review",
        "reported by monitoring teams"
    ],
    "Insider Misuse": [
        "identified by access control review",
        "reported by internal teams",
        "detected during privilege audits",
        "discovered during policy enforcement checks"
    ],
    "Denial of Service": [
        "identified by network monitoring tools",
        "reported by service owners",
        "detected after service outages",
        "observed during traffic analysis"
    ]
}

IMPACT = {
    "Malware": [
        "potential data exposure",
        "risk of lateral movement",
        "system instability",
        "credential compromise"
    ],
    "Ransomware": [
        "data loss",
        "extended system downtime",
        "business disruption",
        "operational paralysis"
    ],
    "Phishing": [
        "credential compromise",
        "unauthorized account access",
        "risk of further attacks",
        "identity misuse"
    ],
    "Data Breach": [
        "exposure of sensitive information",
        "regulatory non-compliance",
        "loss of customer trust",
        "legal and financial impact"
    ],
    "Insider Misuse": [
        "policy violations",
        "unauthorized data access",
        "internal security risk",
        "loss of data integrity"
    ],
    "Denial of Service": [
        "service disruption",
        "customer impact",
        "availability loss",
        "business continuity issues"
    ]
}
# TEMPLATES = {
#     "Malware": [
#         "A critical {obs} was {det}, signaling a potential infection. This creates a risk of {imp}.",
#         "Security alert: {det} identifying {obs}. Immediate action required to prevent {imp}.",
#         "{obs} detected via {det}. Primary concern: {imp}.",
#         "Investigation initialized after {det} reported {obs}. The threat could lead to {imp}.",
#         "System diagnostics: {obs}. Detection source: {det}. Forecasted impact: {imp}."
#     ],
#     "Ransomware": [
#         "Urgent: {obs} has been {det}. This event threatens {imp} and mandates containment.",
#         "{det} flagged {obs}. We are observing indicators of {imp}.",
#         "Ransomware indicators present: {obs}. Discovery method: {det}. Risk profile: {imp}.",
#         "Systems are compromised with {obs} as {det}. Operations face {imp}."
#     ],
#     "Phishing": [
#         "Phishing attempt confirmed: {obs} was {det}. The primary risk is {imp}.",
#         "User report of {obs}. Analysis via {det} confirms phishing activity. Impact: {imp}.",
#         "Security team alerted to {obs}. This was {det}. We are mitigating against {imp}.",
#         "Alert: Potential {imp} due to {obs}, which was {det}."
#     ],
#     "Data Breach": [
#         "Data breach alert: {obs} was {det}. This has resulted in {imp}.",
#         "Audit trail analysis ({det}) uncovered {obs}. The incident poses a threat of {imp}.",
#         "Critical incident: {obs} identified. Detection method was {det}. Consequence is {imp}.",
#         "Evidence of {obs} was found; it was {det}. This breach could lead to {imp}."
#     ],
#     "Insider Misuse": [
#         "Insider misuse case opened. An individual's {obs} was {det}. This constitutes {imp}.",
#         "{det} revealed {obs}. This is being investigated as a case of {imp}.",
#         "Internal alert for {obs}. The activity was {det}. This is a clear case of {imp}.",
#         "Policy violation: {obs}. The issue was {det}. The impact is {imp}."
#     ],
#     "Denial of Service": [
#         "Denial of Service attack in progress: {obs} was {det}. The main effect is {imp}.",
#         "Network monitoring ({det}) shows {obs}. We are experiencing {imp}.",
#         "Service alert: {obs}. This was {det} and is causing {imp}.",
#         "Traffic analysis confirms {obs}, which was {det}. The result is {imp}."
#     ],
#     "Default": [
#         "Incident Report: {obs} ({det}). Assessing risk of {imp}.",
#         "{det} has logged {obs}, resulting in potential {imp}.",
#         "Security breach indicator: {obs}. Validated by {det}. Risk: {imp}."
#     ]
# }
# 
# # Helper to safely get a template
# def get_template(incident_type):
#     return TEMPLATES.get(incident_type, TEMPLATES["Default"])

# ============================================================
# COMPONENT VARIATIONS (SEMANTIC PARAPHRASING)
# ============================================================

# Instead of 1 string per ID, we use a list of synonymous strings.
OBSERVATIONS_VARIANTS = {
    # Malware
    "Malware_Slowdown": [
        "unusual system slowdown",
        "significant degradation in endpoint performance",
        "users reporting extreme latency",
        "sluggish response times across applications"
    ],
    "Malware_BackgroundProcesses": [
        "unexpected background processes",
        "unrecognized processes consuming system resources",
        "hidden processes detected in the task manager",
        "anomalous services starting automatically"
    ],
    "Malware_HighCPU": [
        "high CPU usage by unknown executables",
        "unidentified executables causing CPU spikes",
        "processor utilization is maxed out by a strange process",
        "an unknown binary is consuming excessive CPU cycles"
    ],
    "Malware_AppCrashes": [
        "frequent application crashes",
        "multiple applications are terminating unexpectedly",
        "instability in core business applications",
        "programs closing without warning"
    ],
    "Malware_UnauthExecution": [
        "unauthorized software execution",
        "software running on a system without approval",
        "an unapproved application was executed",
        "detection of blacklisted software running on an endpoint"
    ],
    # Ransomware
    "Ransomware_EncryptedFiles": [
        "encrypted files across multiple directories",
        "files have been rendered unreadable by encryption",
        "widespread file encryption detected on a server",
        "user data has been systematically encrypted"
    ],
    "Ransomware_InaccessibleData": [
        "inaccessible system data",
        "critical data can no longer be accessed",
        "users are unable to open or read important files",
        "system data has become unavailable"
    ],
    "Ransomware_UnauthEncryption": [
        "unauthorized encryption activity",
        "detection of a process rapidly encrypting files",
        "anomalous file modification activity consistent with encryption",
        "a non-standard process is performing cryptographic operations"
    ],
    "Ransomware_FileAccessLoss": [
        "sudden loss of file access",
        "users abruptly lost access to their files",
        "permissions on numerous files were changed, denying access",
        "an immediate inability to access network shares"
    ],
    "Ransomware_RansomNotes": [
        "appearance of ransom notes",
        "text files containing ransom demands found on desktops",
        "discovery of files with instructions on how to pay a ransom",
        "ransom notes have been placed in multiple directories"
    ],
    # Phishing
    "Phishing_CredentialEntry": [
        "users entering credentials on suspicious links",
        "employees have submitted their login details on a fake portal",
        "credential harvesting detected from a phishing site",
        "user credentials were captured by a fraudulent login page"
    ],
    "Phishing_DeceptiveEmails": [
        "employees reporting deceptive emails",
        "multiple users have forwarded suspicious emails to IT security",
        "a wave of phishing emails has been reported by staff",
        "reports from employees about emails impersonating leadership"
    ],
    "Phishing_UnexpectedPrompts": [
        "unexpected login prompts",
        "users are seeing unusual authentication requests",
        "an application is prompting for credentials at an odd time",
        "spurious login prompts appearing for multiple users"
    ],
    "Phishing_FakeWebsiteSubmission": [
        "credential submission on fake websites",
        "evidence of users logging into a typosquatted domain",
        "traffic detected to a known phishing website",
        "a fake website successfully harvested user credentials"
    ],
    "Phishing_EmailCampaigns": [
        "suspicious email campaigns",
        "a coordinated phishing campaign targeting the finance department",
        "detection of a large-scale, malicious email blast",
        "an ongoing email campaign with malicious attachments"
    ],
    # Data Breach
    "DataBreach_DbQueries": [
        "unauthorized database queries",
        "suspicious SQL queries executed against the customer database",
        "an application account running unexpected database commands",
        "detection of anomalous queries to sensitive tables"
    ],
    "DataBreach_SensitiveRecords": [
        "unexpected access to sensitive records",
        "an account accessed PII records outside of normal business hours",
        "unwarranted access to confidential customer data",
        "logs show access to sensitive files that was not authorized"
    ],
    "DataBreach_AbnormalAccess": [
        "abnormal data access patterns",
        "a user account is accessing data in a highly unusual sequence",
        "anomalous data access patterns detected by monitoring tools",
        "behavioral analytics flagged unusual data access"
    ],
    "DataBreach_DataExfiltration": [
        "large volumes of data extraction",
        "detection of a significant amount of data being exfiltrated from the network",
        "an unusually large data transfer to an external IP address",
        "evidence of bulk data export from a critical database"
    ],
    "DataBreach_UnfamiliarLocation": [
        "access from unfamiliar locations",
        "logins detected from a geographically anomalous location",
        "a user account authenticated from a country we do not operate in",
        "access attempts from an IP address on a threat intelligence blacklist"
    ],
    # Insider Misuse
    "Insider_PrivilegedMisuse": [
        "misuse of privileged system access",
        "an administrator account was used for non-administrative purposes",
        "abuse of root/admin privileges detected",
        "a privileged account performed actions outside of its role"
    ],
    "Insider_InternalDataAccess": [
        "unauthorized internal data access",
        "an employee accessed files not related to their job function",
        "internal audit flagged an employee accessing sensitive HR records",
        "a user was found browsing confidential project folders without need-to-know"
    ],
    "Insider_PolicyViolation": [
        "policy violations by internal users",
        "an employee violated the acceptable use policy",
        "detection of internal user activity that contravenes security policy",
        "a user bypassed security controls in violation of company policy"
    ],
    "Insider_BeyondResponsibilities": [
        "access beyond job responsibilities",
        "an employee's data access patterns exceed their job role",
        "a user is accessing data that is not required for their duties",
        "detection of access to data inconsistent with the user's department"
    ],
    "Insider_RestrictedAccess": [
        "repeated access to restricted resources",
        "an employee made multiple attempts to access a restricted server",
        "a user account is repeatedly trying to access confidential files",
        "logs show persistent access attempts to a secured resource by an internal user"
    ],
    # Denial of Service
    "DoS_ServiceUnavailability": [
        "service unavailability due to traffic spikes",
        "the public website is down following a massive traffic surge",
        "a sudden increase in traffic has rendered the service unresponsive",
        "customer-facing services are offline due to an overwhelming number of requests"
    ],
    "DoS_Congestion": [
        "network congestion from excessive requests",
        "a massive spike in inbound traffic causing bottlenecks",
        "saturation of network bandwidth due to high request volume",
        "unnatural traffic patterns flooding the gateway"
    ],
    "DoS_SystemDowntime": [
        "system downtime caused by overload",
        "servers have gone offline due to resource exhaustion",
        "an application server has crashed due to being overloaded",
        "critical systems are down as a result of a resource overload"
    ],
    "DoS_ResourceExhaustion": [
        "resource exhaustion on network devices",
        "firewalls and load balancers are reporting 100% CPU utilization",
        "network infrastructure is failing due to resource depletion",
        "critical network devices are running out of memory and dropping packets"
    ],
    "DoS_DegradedPerformance": [
        "degraded service performance",
        "application response times have increased dramatically",
        "users are reporting extreme slowness and timeouts",
        "the service is still online but performance is severely impacted"
    ]
}

# Helper to grab a random variation
def get_variant(key, variant_map):
    # In a real app, you'd map the Incident Type to these keys.
    # For now, let's assume you restructure your OBSERVATIONS dict 
    # to hold lists of lists, or use a key-based lookup.
    return random.choice(variant_map.get(key, ["Generic error"]))




# replacing templates logic
def generate_dynamic_description(incident_type, obs, det, imp):
    """
    Constructs a sentence by randomly choosing an information hierarchy
    and phrasing style.
    """
    
    # 4 Distinct Architectures for the sentence
    structures = [
        # Structure A: Standard (Obs -> Det -> Imp)
        lambda o, d, i: f"Incident Log: {o} was identified via {d}. This has resulted in {i}.",
        
        # Structure B: Action-First (Det -> Obs -> Imp)
        lambda o, d, i: f"{d} flagged {o}, posing an immediate risk of {i}.",
        
        # Structure C: Impact-Centric (Imp -> Obs) - Hides detection sometimes or moves it
        lambda o, d, i: f"Critical alert: Risk of {i} detected. Root cause appears to be {o} as noted by {d}.",
        
        # Structure D: Concise / Telegraphic (Good for variety)
        lambda o, d, i: f"SECURITY EVENT: {o} // SOURCE: {d} // IMPACT: {i}"
    ]
    
    # Select structure
    structure_func = random.choice(structures)
    
    # Return the assembled string
    return structure_func(obs, det, imp)
# ============================================================
# UPDATED: BUILD DESCRIPTION POOL
# ============================================================

def build_description_pool(incident_type):
    pool = []
    
    # We loop through your existing lists (assuming they are strings).
    # Ideally, you convert your lists to the "Synonym Injection" format above.
    # But even with your CURRENT lists, "Slot Shuffling" will reduce similarity to ~0.70-0.80.
    
    for obs, det, imp in product(
        OBSERVATIONS[incident_type],
        DETECTION[incident_type],
        IMPACT[incident_type]
    ):
        # Generate 3 variations for EVERY combination to ensure we have choice
        # This increases the pool size artificially but adds diversity.
        for _ in range(3):
            desc = generate_dynamic_description(incident_type, obs, det, imp)
            pool.append(desc)
            
    # Remove exact duplicates just in case
    pool = list(set(pool))
    random.shuffle(pool)
    return pool

DESCRIPTION_POOLS = {
    itype: build_description_pool(itype) for itype in INCIDENT_TYPES
}

# Safety check
for itype, pool in DESCRIPTION_POOLS.items():
    if len(pool) < INCIDENTS_PER_TYPE:
        raise RuntimeError(
            f"Description pool too small for {itype}: "
            f"{len(pool)} < {INCIDENTS_PER_TYPE}"
        )

# ============================================================
# OUTCOME DISTRIBUTION (REALISTIC)
# ============================================================

OUTCOME_POOL = (
    ["Success"] * 18 +
    ["Partial"] * 9 +
    ["Failure"] * 3
)

# ============================================================
# INCIDENT GENERATOR
# ============================================================

def generate_incident(incident_type):
    return {
        "incident_id": f"INC-{uuid4().hex[:8]}",
        "incident_title": f"{incident_type} incident reported",
        "incident_description": DESCRIPTION_POOLS[incident_type].pop(),
        "incident_type": incident_type,
        "asset_type": ASSET_MAP[incident_type],
        "business_impact": BUSINESS_IMPACT_MAP[incident_type],
        "user_privilege": random.choice(["User", "Admin"]),
        "recommended_action_ids": ";".join(ACTION_LIBRARY[incident_type]),
        "outcome": random.choice(OUTCOME_POOL)
    }

# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(no_of_cases=30, output_file="incidents.csv"):
    records = []

    for incident_type in INCIDENT_TYPES:
        for _ in range(INCIDENTS_PER_TYPE):
            records.append(generate_incident(incident_type))

    random.shuffle(records)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"[+] Generated {TOTAL_INCIDENTS} incidents (balanced, unique text)")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_dataset(no_of_cases=45, output_file="data/incidents.csv")
