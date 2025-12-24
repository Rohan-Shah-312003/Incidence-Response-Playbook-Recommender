import pandas as pd
import re
from email import policy
from email.parser import Parser

RAW_ENRON_PATH = "./data/enron_raw/enron_emails.csv"
OUTPUT_PATH = "./data/enron_processed/enron_incidents.csv"

# CLEAN TEXT
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# EXTRACT EMAIL BODY FROM RAW MESSAGE
def extract_body(raw_message):
    try:
        email = Parser(policy=policy.default).parsestr(raw_message)

        if email.is_multipart():
            parts = [
                part.get_payload(decode=True)
                for part in email.walk()
                if part.get_content_type() == "text/plain"
            ]
            body = b" ".join(p for p in parts if p)
        else:
            body = email.get_payload(decode=True)

        return body.decode(errors="ignore") if body else ""

    except Exception:
        return ""

# WEAK LABELING
def assign_label(text):
    phishing_terms = [
        "password", "login", "verify", "click", "credentials", "urgent"
    ]
    breach_terms = [
        "confidential", "restricted", "sensitive", "private", "do not share"
    ]
    insider_terms = [
        "internal use only", "unauthorized", "policy violation",
        "not approved", "without permission"
    ]

    if any(t in text for t in phishing_terms):
        return "Phishing"
    if any(t in text for t in breach_terms):
        return "Data Breach"
    if any(t in text for t in insider_terms):
        return "Insider Misuse"

    return None

# MAIN PIPELINE
def preprocess_enron():
    df = pd.read_csv(RAW_ENRON_PATH)

    if "message" not in df.columns:
        raise RuntimeError(f"'message' column not found. Columns: {df.columns}")

    records = []

    for raw_msg in df["message"].dropna():
        body = extract_body(raw_msg)
        body = clean_text(body)

        if len(body) < 80:
            continue

        label = assign_label(body)
        if label:
            records.append({
                "text": body,
                "incident_type": label,
                "source": "ENRON"
            })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"[+] Saved {len(out_df)} Enron incident texts to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_enron()
