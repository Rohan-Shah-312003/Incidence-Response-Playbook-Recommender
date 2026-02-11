### 1. Malware

Automated endpoint alerts flagged a suspicious execution of `certutil.exe` on a workstation in the Engineering department, which was used to download a binary from an unidentified external IP. The process subsequently attempted to inject code into `explorer.exe` and established a persistent outbound connection via port 443. No authorized software updates or developer activities were scheduled for this host, indicating a high probability of a live Trojan infection and active command-and-control (C2) beaconing.

### 2. Insider Misuse

A Junior Administrator was observed using a domain admin service account to perform recursive directory searches on a restricted "Legal_Counsel" file share. Logs indicate the user attempted to bypass standard RBAC by temporarily adding their personal account to a high-privilege security group during a period of low staffing. This activity does not align with any open support tickets or documented infrastructure maintenance, suggesting intentional privilege escalation for unauthorized data access.

### 3. Data Breach --> is giving malware as o/p

Security monitoring detected a massive spike in outbound HTTPS traffic from a production database server to an unfamiliar cloud storage bucket. Initial analysis confirms the exfiltration of approximately 50,000 records containing customer PII, including names and encrypted tax IDs, via a compromised API key. There were no approved data migrations or integration tests active during this window, representing a confirmed breach of sensitive organizational data.

### 4. Phishing

An employee reported a "Security Alert" email that requested an urgent password update via a link to a cloned corporate login page. Shortly after the email was opened, multiple failed login attempts were recorded for that user from a geo-location inconsistent with their current travel status, eventually triggering an account lockout. The event demonstrates a successful credential harvesting attempt with subsequent unauthorized access efforts.

### 5. Ransomware

System monitors on a core file server alerted to a sudden, high-frequency rename of files to a `.crypt` extension and the mass deletion of Volume Shadow Copies. A text-based ransom note was deposited in every affected directory, providing instructions for decryption via a dark-web portal. The rapid encryption of shared drives and the disabling of local recovery points indicate an active, automated ransomware deployment requiring immediate isolation.

### 6. DoS

The external-facing web portal became unresponsive following a surge of specialized HTTP POST requests designed to exhaust the application's database connection pool. Traffic analysis shows the requests originated from a globally distributed botnet, bypassing standard volumetric firewalls by targeting specific resource-heavy application functions. No legitimate traffic spikes or marketing events were planned, confirming a targeted Layer 7 Denial of Service attack.
