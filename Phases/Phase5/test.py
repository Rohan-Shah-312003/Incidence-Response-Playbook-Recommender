from core.classifier import classify_incident

# Test Malware detection
malware_test = """
Endpoint protection detected suspicious executable svchost32.exe running 
with elevated privileges. Process attempted to modify system registry keys.
"""
label, conf = classify_incident(malware_test)
print(f"Malware test: {label} ({conf:.2%})")

# Test Ransomware detection  
ransomware_test = """
Critical alert: Rapid file encryption detected on file server SRV-FILE-01.
Multiple directories showing .locked extension. Ransom note present demanding payment.
"""
label, conf = classify_incident(ransomware_test)
print(f"Ransomware test: {label} ({conf:.2%})")

# Test DoS detection
dos_test = """
Network operations center reports severe bandwidth saturation. Traffic analysis 
shows distributed denial of service attack from 10,000 source IPs.
"""
label, conf = classify_incident(dos_test)
print(f"DoS test: {label} ({conf:.2%})")