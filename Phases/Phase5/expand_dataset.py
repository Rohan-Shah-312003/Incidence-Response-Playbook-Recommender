"""
Expand dataset to include Malware, Ransomware, and DoS incidents
This creates synthetic training data for the missing incident types.
"""

import pandas as pd
import numpy as np
from uuid import uuid4
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# ================================================================
# INCIDENT TEMPLATES FOR MISSING TYPES
# ================================================================

MALWARE_TEMPLATES = [
    "Endpoint protection detected suspicious executable {} running with elevated privileges. Process attempted to modify system registry keys and establish persistence mechanisms.",
    "Antivirus scan identified malicious binary {} in user downloads folder. File exhibits behavior consistent with trojan activity including keylogging and screenshot capture.",
    "Security monitoring flagged unusual process {} consuming excessive CPU resources. Analysis reveals code injection attempts and unauthorized network connections to C2 server.",
    "Behavioral analysis detected malware {} attempting to disable Windows Defender and modify firewall rules. Process shows signs of rootkit installation.",
    "User reported system slowdown. Investigation found unknown process {} exfiltrating data and communicating with known malicious domain.",
    "EDR solution quarantined suspicious file {} exhibiting polymorphic malware characteristics. Binary contains obfuscated payload and anti-analysis techniques.",
    "Automated scan discovered trojan {} embedded in PDF attachment. File attempts privilege escalation and creates scheduled task for persistence.",
    "Memory forensics revealed malware {} injected into legitimate system process. Malicious code establishes reverse shell and downloads additional payloads.",
    "Threat intelligence match identified known malware family {} on workstation. Process demonstrates lateral movement attempts and credential harvesting.",
    "Security tools blocked execution of {} dropper attempting to install cryptocurrency miner. File exploits system vulnerabilities for privilege escalation.",
    "Analysis of compromised system shows malware {} modified hosts file and installed browser extensions for ad injection and credential theft.",
    "Network monitoring detected malware {} beaconing to command and control infrastructure at regular intervals using encrypted channel.",
    "Incident response identified fileless malware {} running in memory without persistent storage footprint. Payload delivered via PowerShell script.",
    "Sandbox detonation of suspicious email attachment revealed malware {} with remote access trojan capabilities and keylogging functionality.",
    "Forensic investigation uncovered malware {} utilizing DLL injection to evade detection and maintain persistence across system reboots.",
]

RANSOMWARE_TEMPLATES = [
    "Critical alert: Rapid file encryption detected on file server {}. Multiple directories showing .{} extension added to files. Ransom note present demanding cryptocurrency payment.",
    "Mass file modification event on shared drive {}. Encryption process consuming maximum CPU. Backup deletion attempts detected. Ransom demand displayed on affected systems.",
    "Database server {} experiencing widespread data encryption. Shadow copies deleted. Ransom message indicates {} variant with 72-hour payment deadline.",
    "Network shares on server {} being encrypted in real-time. File integrity monitoring triggered thousands of alerts. Desktop wallpaper replaced with payment instructions.",
    "Disaster recovery team alerted to encryption of production data on {}. Investigation reveals {} ransomware family. Attackers threatening data publication if ransom unpaid.",
    "Emergency response: File server {} locked by encryption malware. All business-critical documents inaccessible. Decryption key held hostage with Bitcoin ransom demand.",
    "Automated backup failed due to encrypted source files on {}. Analysis confirms {} ransomware deployment. Lateral movement detected across network segments.",
    "Users unable to access files on {}. Forensics reveal systematic encryption using strong algorithm. Ransom note references recent data breach and threatens exposure.",
    "Containment initiated after ransomware {} identified encrypting virtual machine images. Snapshot deletion detected. Attackers demand payment via anonymous cryptocurrency.",
    "Server {} exhibiting signs of double extortion ransomware. Files encrypted and copied to attacker infrastructure before encryption. Leak site updated with victim details.",
    "Critical incident: Production database on {} encrypted by {} ransomware. Exfiltration detected prior to encryption. Ransom negotiation initiated by attackers via TOR.",
    "File encryption spreading across network from patient zero at {}. Ransomware {} propagating via SMB shares. Imminent data loss risk if containment fails.",
    "Backup server {} compromised before encryption event. Ransomware deleted volume shadow copies and disabled restore points. Payment portal provided in ransom message.",
    "Email server {} encrypted by {} variant. Mailbox data inaccessible. Attackers claim data backup and threaten GDPR notification if ransom not paid within 48 hours.",
    "Engineering file server {} hit by ransomware encryption routine. CAD files and intellectual property affected. Decryption testing shows strong encryption requiring private key.",
]

DOS_TEMPLATES = [
    "Network operations center reports severe bandwidth saturation on edge router {}. Traffic analysis shows distributed denial of service attack from {} source IPs.",
    "Web application {} experiencing service degradation. Layer 7 DDoS attack detected with {} requests per second overwhelming application logic and database queries.",
    "Infrastructure monitoring alerts on {} due to connection table exhaustion. SYN flood attack consuming all available TCP connections and preventing legitimate access.",
    "Public-facing API {} returning timeout errors. Traffic analysis reveals amplification attack utilizing {} reflection. Bandwidth exceeded maximum capacity by 300%.",
    "Customer complaints of service unavailability for {}. Investigation confirms volumetric DDoS attack generating {} Gbps traffic, exceeding ISP capacity limits.",
    "Load balancer {} failing health checks due to resource exhaustion. UDP flood attack detected with {} packets per second targeting DNS services.",
    "Email gateway {} experiencing delivery delays. SMTP flood attack overwhelming mail queuing system with {} connections attempting simultaneous delivery.",
    "E-commerce platform {} down during peak sales period. Distributed attack from botnet sources targeting checkout process with malformed HTTP requests.",
    "VoIP infrastructure {} suffering call quality degradation. Analysis shows RTP flood attack consuming bandwidth and causing packet loss exceeding acceptable thresholds.",
    "Database cluster {} experiencing query timeout failures. Application-layer attack exploiting expensive queries to exhaust connection pool and CPU resources.",
    "CDN reporting cache bypass attacks on {}. Attackers generating unique URLs to prevent caching and overwhelm origin servers with {} requests per minute.",
    "Firewall {} dropping legitimate traffic due to state table overflow. Attack leveraging fragmented packets to consume connection tracking resources.",
    "Authentication service {} unavailable due to credential stuffing attack. {} login attempts per second exhausting rate limiting and locking out legitimate users.",
    "Video streaming platform {} buffering for all users. Bandwidth exhaustion attack targeting edge servers with {} Mbps sustained traffic from IoT botnet.",
    "Financial services portal {} experiencing transaction failures. Low-and-slow HTTP attack maintaining {} connections to exhaust server thread pools.",
]

# ================================================================
# ARTIFACT NAMES FOR VARIETY
# ================================================================

MALWARE_ARTIFACTS = [
    "svchost32.exe", "updater.dll", "chrome_update.exe", "system32.scr",
    "java_installer.msi", "flash_player.exe", "winlogon.dll", "explorer32.exe",
    "nvidia_driver.sys", "office_update.exe", "rundll64.exe", "taskhost.exe",
    "csrss32.exe", "lsass64.exe", "smss.scr", "services32.exe"
]

RANSOMWARE_FAMILIES = [
    "LockBit", "BlackCat", "Royal", "Akira", "BlackMatter", "Conti",
    "REvil", "Ryuk", "Maze", "DarkSide", "Hive", "Cuba", "Vice Society"
]

RANSOMWARE_EXTENSIONS = [
    "locked", "encrypted", "locked2023", "CRYPTED", "enc", "crypt",
    "lock", "encrypted2023", "secure", "locked_files"
]

SERVERS = [
    "SRV-FILE-01", "SRV-DB-PROD", "SRV-WEB-02", "SRV-APP-03",
    "SRV-SHARE-FINANCE", "SRV-BACKUP-01", "SRV-EMAIL-EXCH",
    "SRV-SQL-PROD", "SRV-VM-HOST-01", "SRV-DC-PRIMARY"
]

# ================================================================
# GENERATION FUNCTION
# ================================================================

def generate_synthetic_incidents(samples_per_type=3000):
    """
    Generate synthetic incidents for Malware, Ransomware, and DoS
    
    Args:
        samples_per_type: Number of samples to generate per incident type
    
    Returns:
        DataFrame with synthetic incident data
    """
    records = []
    
    # Generate Malware incidents
    print(f"Generating {samples_per_type} Malware incidents...")
    for i in range(samples_per_type):
        template = random.choice(MALWARE_TEMPLATES)
        artifact = random.choice(MALWARE_ARTIFACTS)
        text = template.format(artifact)
        
        records.append({
            'text': text,
            'incident_type': 'Malware',
            'source': 'SYNTHETIC'
        })
    
    # Generate Ransomware incidents
    print(f"Generating {samples_per_type} Ransomware incidents...")
    for i in range(samples_per_type):
        template = random.choice(RANSOMWARE_TEMPLATES)
        server = random.choice(SERVERS)
        family = random.choice(RANSOMWARE_FAMILIES)
        ext = random.choice(RANSOMWARE_EXTENSIONS)
        
        # Randomly use family or extension
        if random.random() > 0.5:
            text = template.format(server, family)
        else:
            text = template.format(server, ext)
        
        records.append({
            'text': text,
            'incident_type': 'Ransomware',
            'source': 'SYNTHETIC'
        })
    
    # Generate Denial of Service incidents
    print(f"Generating {samples_per_type} Denial of Service incidents...")
    for i in range(samples_per_type):
        template = random.choice(DOS_TEMPLATES)
        server = random.choice(SERVERS)
        volume = random.choice([
            '10,000', '50,000', '100,000', '500,000', '1M',
            '5 Gbps', '10 Gbps', '50 Gbps', '100 Gbps',
            '1,000', '5,000', '25,000'
        ])
        
        text = template.format(server, volume)
        
        records.append({
            'text': text,
            'incident_type': 'Denial of Service',
            'source': 'SYNTHETIC'
        })
    
    df = pd.DataFrame(records)
    print(f"\n✓ Generated {len(df)} synthetic incidents")
    return df


def merge_with_existing(synthetic_df, existing_csv_path, output_path):
    """
    Merge synthetic data with existing real incidents
    
    Args:
        synthetic_df: DataFrame with synthetic incidents
        existing_csv_path: Path to existing real_incidents_balanced.csv
        output_path: Where to save merged dataset
    """
    try:
        # Load existing data
        print(f"\nLoading existing data from {existing_csv_path}...")
        existing_df = pd.read_csv(existing_csv_path)
        print(f"  Existing samples: {len(existing_df)}")
        print(f"  Existing types: {existing_df['incident_type'].unique().tolist()}")
        
        # Merge
        merged_df = pd.concat([existing_df, synthetic_df], ignore_index=True)
        
        # Shuffle
        merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        merged_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Merged dataset saved to {output_path}")
        print(f"\nFinal dataset distribution:")
        print(merged_df['incident_type'].value_counts())
        print(f"\nTotal samples: {len(merged_df)}")
        
        return merged_df
        
    except FileNotFoundError:
        print(f"\n⚠️  Existing file not found at {existing_csv_path}")
        print("Saving synthetic data only...")
        
        synthetic_df.to_csv(output_path, index=False)
        print(f"✓ Synthetic dataset saved to {output_path}")
        print(f"\nDataset distribution:")
        print(synthetic_df['incident_type'].value_counts())
        
        return synthetic_df


if __name__ == "__main__":
    # Generate synthetic data
    synthetic_df = generate_synthetic_incidents(samples_per_type=500)
    
    # Define paths (adjust these to match your project structure)
    EXISTING_DATA = "./data/real_incidents_balanced.csv"
    OUTPUT_PATH = "./data/real_incidents_expanded.csv"
    
    # Merge with existing data
    final_df = merge_with_existing(synthetic_df, EXISTING_DATA, OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("✓ DATASET EXPANSION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the expanded dataset at:", OUTPUT_PATH)
    print("2. Update training scripts to use: real_incidents_expanded.csv")
    print("3. Retrain models with: python train_evaluate_pipeline.py")