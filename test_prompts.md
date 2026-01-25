**Prompt** - During routine monitoring, the SOC team observed repeated authentication attempts on a file server at approximately 12:30 AM, outside normal business hours. The attempts originated from a valid internal user account that is not authorized to access the server. Shortly after, several unusual log entries were generated indicating access to sensitive directories. No scheduled maintenance or automated jobs were planned at this time.

**Override** - Assume the user account has no business justification for after-hours access and treat this as a potential compromised credential rather than misconfiguration.
Prioritize analysis for lateral movement, credential abuse, and privilege escalation, even if indicators appear subtle.

**Prompt** - An employee reported receiving an urgent email late at night claiming to be from the IT department, requesting immediate password verification to avoid account suspension. The email contained a link redirecting to an external website resembling the corporate login portal. Shortly after the report, failed login attempts were observed on the employee’s account.

**Override** - Treat the email as confirmed phishing and assume the user may have entered credentials, regardless of user denial.
Focus on account compromise risk, downstream access attempts, and mailbox-based propagation, not just email classification.

**Prompt** Automated alerts indicated outbound data transfers from a database server to an unfamiliar external IP address during early morning hours. The server is typically idle at this time. Initial review shows that the transfers were initiated using a service account with elevated privileges. No recent configuration changes were documented.

**Override** - Assume the service account activity is unauthorized and not part of any backup, replication, or vendor integration.
Prioritize data exfiltration analysis, including scope of data accessed, transfer volume, and persistence mechanisms.


**Prompt** - Unrecognized authentication attempts and sensitive file access were observed on an internal server after midnight using a valid employee account, with no scheduled activity planned.

**Override** - Do not assume insider intent or authorized overtime.
Analyze this as a credential misuse scenario with emphasis on initial access vector, session hijacking, and abnormal authentication patterns.

**Prompt** - During routine monitoring, repeated authentication attempts were observed on a file server at 12:30 AM. The attempts originated from a valid internal user account accessing sensitive directories.

**Override** - Treat repeated authentication attempts as deliberate credential probing, not user error.
Emphasize brute-force behavior, password spraying, and account enumeration, even if the account is internal and valid.