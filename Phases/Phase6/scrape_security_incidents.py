"""
Scrape real-world malware and ransomware incident descriptions
from public security blogs and news sites

This provides actual incident narratives instead of just IoCs
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import re

# Ethical scraping with delays
DELAY_SECONDS = 2


def scrape_bleepingcomputer_malware(max_pages=5):
    """
    Scrape malware incident descriptions from BleepingComputer

    Args:
        max_pages: Number of pages to scrape

    Returns:
        List of incident dictionaries
    """
    incidents = []
    base_url = "https://www.bleepingcomputer.com/tag/malware/"

    print("Scraping BleepingComputer malware news...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for page in range(1, max_pages + 1):
        url = f"{base_url}page/{page}/" if page > 1 else base_url

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Find article summaries
            articles = soup.find_all("article", class_="bc_latest_news_text")

            for article in articles:
                try:
                    # Extract text
                    summary = article.find("p", class_="bc_excerpt")
                    if summary:
                        text = summary.get_text(strip=True)

                        # Basic filtering
                        if len(text) > 50:
                            incidents.append(
                                {
                                    "text": text,
                                    "incident_type": "Malware",
                                    "source": "BleepingComputer",
                                }
                            )

                except Exception as e:
                    continue

            print(f"  Page {page}: {len(articles)} articles")
            time.sleep(DELAY_SECONDS)

        except Exception as e:
            print(f"Error on page {page}: {e}")
            break

    print(f"✓ Scraped {len(incidents)} malware incidents\n")
    return incidents


def scrape_bleepingcomputer_ransomware(max_pages=5):
    """
    Scrape ransomware incident descriptions
    """
    incidents = []
    base_url = "https://www.bleepingcomputer.com/tag/ransomware/"

    print("Scraping BleepingComputer ransomware news...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for page in range(1, max_pages + 1):
        url = f"{base_url}page/{page}/" if page > 1 else base_url

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.find_all("article", class_="bc_latest_news_text")

            for article in articles:
                try:
                    summary = article.find("p", class_="bc_excerpt")
                    if summary:
                        text = summary.get_text(strip=True)

                        if len(text) > 50:
                            incidents.append(
                                {
                                    "text": text,
                                    "incident_type": "Ransomware",
                                    "source": "BleepingComputer",
                                }
                            )

                except Exception as e:
                    continue

            print(f"  Page {page}: {len(articles)} articles")
            time.sleep(DELAY_SECONDS)

        except Exception as e:
            print(f"Error on page {page}: {e}")
            break

    print(f"✓ Scraped {len(incidents)} ransomware incidents\n")
    return incidents


def generate_synthetic_incidents_from_reports(category, num_samples=100):
    """
    Generate synthetic but realistic incidents using LLM
    based on real attack patterns

    This is a template - you would use Groq/OpenAI API here

    Args:
        category: 'Malware', 'Ransomware', or 'Denial of Service'
        num_samples: Number to generate
    """
    print(f"Generating {num_samples} synthetic {category} incidents...")

    # Templates based on real patterns
    templates = {
        "Malware": [
            "Security team detected unusual process executing on endpoint {host}, identified as {malware_family} variant attempting to establish command and control communication.",
            "User reported system slowdown. Investigation revealed {malware_type} infection spreading through network share, affecting {count} workstations.",
            "Automated scan discovered {malware_family} trojan on server {server_name}, exfiltrating data to external IP address over encrypted channel.",
            "Endpoint protection blocked execution of suspicious binary {hash}, analysis confirmed it as {malware_type} attempting privilege escalation.",
            "Multiple hosts showing signs of {malware_family} infection including registry modifications, scheduled tasks creation, and network scanning behavior.",
        ],
        "Ransomware": [
            "File server experienced mass encryption event affecting {volume} of data, ransom note identified as {ransomware_family} demanding payment.",
            "User reported inability to access files with .{extension} extension, investigation confirmed {ransomware_family} ransomware deployment.",
            "Backup system detected {count} snapshots deleted in rapid succession, followed by file encryption activity attributed to {ransomware_family}.",
            "Database server encrypted by {ransomware_family}, ransom demand discovered in {location}, operations critically impacted.",
            "Shadow copies deleted across domain, followed by widespread file encryption consistent with {ransomware_family} attack pattern.",
        ],
        "Denial of Service": [
            "Network operations center observed traffic spike to {volume} Gbps targeting public web services, identified as volumetric DDoS attack.",
            "Application performance degraded severely due to sustained request flood from botnet, legitimate users unable to access services.",
            "DNS infrastructure overwhelmed by amplification attack generating {volume} queries per second, causing service outage.",
            "Layer 7 application attack detected targeting login endpoint with {count} requests per second, exhausting server resources.",
            "SYN flood attack consumed all available connections on edge routers, blocking legitimate traffic for {duration} hours.",
        ],
    }

    incidents = []

    # Realistic values for templates
    malware_families = [
        "Emotet",
        "TrickBot",
        "Cobalt Strike",
        "Qakbot",
        "IcedID",
        "BumbleBee",
    ]
    ransomware_families = [
        "LockBit",
        "BlackCat",
        "Royal",
        "Play",
        "Akira",
        "Black Basta",
    ]
    malware_types = ["backdoor", "trojan", "worm", "RAT", "infostealer", "loader"]

    import random

    for i in range(num_samples):
        template = random.choice(templates[category])

        # Fill in placeholders
        text = template.format(
            host=f"WS-{random.randint(1000, 9999)}",
            server_name=f"SRV-{random.choice(['FILE', 'DB', 'APP'])}-{random.randint(10, 99)}",
            malware_family=random.choice(malware_families),
            ransomware_family=random.choice(ransomware_families),
            malware_type=random.choice(malware_types),
            hash=f"{random.randint(1000000, 9999999):x}...",
            count=random.randint(10, 500),
            volume=f"{random.randint(10, 500)} GB"
            if category == "Ransomware"
            else f"{random.randint(50, 500)}",
            extension=random.choice(["locked", "encrypted", "crypt", "locked3"]),
            location="C:\\README.txt",
            duration=random.randint(2, 12),
        )

        incidents.append(
            {"text": text, "incident_type": category, "source": "Synthetic-Enhanced"}
        )

    print(f"✓ Generated {len(incidents)} incidents\n")
    return incidents


def main():
    """
    Main workflow to gather incident data
    """
    print("=" * 70)
    print("INCIDENT DATA COLLECTION")
    print("=" * 70)
    print()

    all_incidents = []

    # Option 1: Scrape real incidents (commented out to avoid actual scraping)
    print("Method 1: Web scraping (use cautiously)")
    all_incidents.extend(scrape_bleepingcomputer_malware(max_pages=3))
    all_incidents.extend(scrape_bleepingcomputer_ransomware(max_pages=3))

    # Option 2: Generate synthetic but realistic incidents
    # print("Method 2: Enhanced synthetic generation")
    # print("-" * 70)
    # all_incidents.extend(generate_synthetic_incidents_from_reports("Malware", 150))
    # all_incidents.extend(generate_synthetic_incidents_from_reports("Ransomware", 150))
    # all_incidents.extend(
    #     generate_synthetic_incidents_from_reports("Denial of Service", 150)
    # )

    # Create DataFrame
    df = pd.DataFrame(all_incidents)

    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"\nTotal incidents collected: {len(df)}")
    print("\nDistribution:")
    print(df["incident_type"].value_counts())

    # Save
    output_path = "./data/scraped_incidents/security_blog_incidents.csv"
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")

    # Show samples
    print("\n" + "=" * 70)
    print("SAMPLE INCIDENTS")
    print("=" * 70)
    for incident_type in df["incident_type"].unique():
        print(f"\n{incident_type}:")
        sample = df[df["incident_type"] == incident_type].iloc[0]["text"]
        print(f"  {sample[:200]}...")


if __name__ == "__main__":
    main()
