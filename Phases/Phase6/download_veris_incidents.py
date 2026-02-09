"""
Download and process VERIS Community Database for incident descriptions

VERIS contains real-world incident narratives across many categories
including malware, ransomware, and DoS attacks.
"""

import requests
import json
import pandas as pd
from pathlib import Path
import re

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent


# VERIS GitHub repository
VERIS_BASE_URL = (
    "https://raw.githubusercontent.com/vz-risk/VCDB/master/data/json/validated/"
)


def download_veris_incidents(output_dir="./data/veris_raw"):
    """
    Download VERIS incident JSON files

    Note: VERIS has hundreds of incident files. We'll need to:
    1. Clone the repo or download via GitHub API
    2. Filter for relevant incident types
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("To download VERIS data, you have two options:")
    print("\n1. Clone the repository:")
    print("   git clone https://github.com/vz-risk/VCDB.git")
    print("\n2. Download the latest release:")
    print("   https://github.com/vz-risk/VCDB/releases")
    print("\nAfter downloading, place JSON files in:", output_path)

    return output_path


def extract_incident_text(incident_json):
    """
    Extract narrative text from VERIS incident JSON

    Args:
        incident_json: Parsed JSON incident object

    Returns:
        Narrative text or None
    """
    # VERIS structure: incident.summary contains narrative
    summary = incident_json.get("summary", "")
    notes = incident_json.get("notes", "")

    # Combine available text
    text_parts = []

    if summary:
        text_parts.append(summary)
    if notes:
        text_parts.append(notes)

    # Also extract from victim/actor descriptions if available
    victim = incident_json.get("victim", {})
    if victim.get("notes"):
        text_parts.append(victim["notes"])

    return " ".join(text_parts) if text_parts else None


def classify_veris_incident(incident_json):
    """
    Classify VERIS incident into our categories

    VERIS uses A4 classification (Actions, Assets, Attributes, Actors)
    We need to map to our 6 categories.

    Args:
        incident_json: Parsed incident

    Returns:
        Our incident type or None
    """
    action = incident_json.get("action", {})
    asset = incident_json.get("asset", {})
    attribute = incident_json.get("attribute", {})

    # Malware indicators
    malware_patterns = action.get("malware", {})
    if malware_patterns.get("variety"):
        malware_types = malware_patterns.get("variety", [])
        # Check for ransomware specifically
        if "Ransomware" in malware_types:
            return "Ransomware"
        return "Malware"

    # Hacking with DoS/DDoS
    hacking = action.get("hacking", {})
    if "DoS" in hacking.get("variety", []) or "DDoS" in hacking.get("variety", []):
        return "Denial of Service"

    # Social engineering -> Phishing
    social = action.get("social", {})
    if "Phishing" in social.get("variety", []):
        return "Phishing"

    # Misuse by internal actors
    misuse = action.get("misuse", {})
    if misuse.get("variety"):
        return "Insider Misuse"

    # Data breach indicators (confidentiality loss)
    confidentiality = attribute.get("confidentiality", {})
    if confidentiality.get("data_disclosure") == "Yes":
        # If not already classified as specific type
        if not any([malware_patterns, hacking, social, misuse]):
            return "Data Breach"

    return None


def process_veris_directory(veris_dir="./VCDB/data/json/validated"):
    """
    Process all VERIS JSON files and extract incident descriptions

    Args:
        veris_dir: Path to VERIS JSON directory

    Returns:
        DataFrame with text and incident_type columns
    """
    veris_path = Path(veris_dir)

    if not veris_path.exists():
        print(f"Error: VERIS directory not found at {veris_path}")
        print("Please clone the repository first:")
        print("  git clone https://github.com/vz-risk/VCDB.git")
        return None

    incidents = []

    print(f"Processing VERIS incidents from {veris_path}...")

    # Process all JSON files
    json_files = list(veris_path.glob("*.json"))
    print(f"Found {len(json_files)} incident files")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                incident = json.load(f)

            # Extract text
            text = extract_incident_text(incident)
            if not text or len(text) < 50:
                continue

            # Classify incident
            incident_type = classify_veris_incident(incident)
            if not incident_type:
                continue

            incidents.append(
                {"text": text, "incident_type": incident_type, "source": "VERIS"}
            )

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    df = pd.DataFrame(incidents)

    print(f"\nExtracted {len(df)} usable incidents:")
    print(df["incident_type"].value_counts())

    return df


def save_processed_veris(output_path="./data/veris_processed/veris_incidents.csv"):
    """
    Main function to process and save VERIS data
    """
    # Process VERIS data
    df = process_veris_directory()

    if df is not None and len(df) > 0:
        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(df)} incidents to {output_file}")

        # Show distribution
        print("\nFinal distribution:")
        print(df.groupby("incident_type").size())

        return df
    else:
        print("No incidents extracted. Check VERIS directory path.")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("VERIS INCIDENT EXTRACTOR")
    print("=" * 60)
    print()

    # First, download/clone VERIS
    print("Step 1: Download VERIS data")
    print("-" * 60)
    download_veris_incidents()

    print("\n" + "=" * 60)
    print("Step 2: Process VERIS incidents")
    print("-" * 60)

    # Then process it
    df = save_processed_veris()

    if df is not None:
        print("\n" + "=" * 60)
        print("✓ VERIS processing complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Merge with existing CERT/Enron data")
        print("2. Supplement with synthetic data if needed")
        print("3. Retrain models")
