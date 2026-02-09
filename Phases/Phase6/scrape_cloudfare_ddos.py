"""
Scrape DDoS incident narratives from Cloudflare blog

Cloudflare publishes detailed DDoS attack reports with:
- Attack descriptions
- Traffic volumes
- Attack vectors
- Mitigation strategies

This is perfect for training your DoS/DDoS classifier.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import re
from pathlib import Path

# Configuration
BASE_URL = "https://blog.cloudflare.com"
DDOS_TAG_URL = "https://blog.cloudflare.com/tag/ddos/"
OUTPUT_PATH = "./data/cloudflare_ddos/cloudflare_ddos_incidents.csv"
DELAY_SECONDS = 3  # Be respectful with scraping

# Headers to mimic browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def get_article_urls(max_pages=5):
    """
    Get list of DDoS-related article URLs from Cloudflare blog

    Args:
        max_pages: Number of tag pages to scrape

    Returns:
        List of article URLs
    """
    article_urls = []

    print(f"Collecting article URLs from {DDOS_TAG_URL}...")

    for page in range(1, max_pages + 1):
        # Cloudflare uses /page/N/ pagination
        if page == 1:
            url = DDOS_TAG_URL
        else:
            url = f"{DDOS_TAG_URL}page/{page}/"

        try:
            print(f"  Page {page}...", end=" ")
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Find article links - Cloudflare blog structure
            # Articles are in divs with specific classes
            articles = soup.find_all("article") or soup.find_all("div", class_="post")

            for article in articles:
                # Find link to full article
                link = article.find("a", href=True)
                if link and link["href"].startswith("/"):
                    full_url = BASE_URL + link["href"]
                    if full_url not in article_urls:
                        article_urls.append(full_url)

            print(f"Found {len(articles)} articles")
            time.sleep(DELAY_SECONDS)

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            break
        except Exception as e:
            print(f"Parsing error: {e}")
            continue

    print(f"\n✓ Collected {len(article_urls)} unique article URLs\n")
    return article_urls


def extract_incident_narratives(article_url):
    """
    Extract DDoS incident descriptions from a single article

    Args:
        article_url: URL of the blog post

    Returns:
        List of incident narrative dictionaries
    """
    incidents = []

    try:
        response = requests.get(article_url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Get article title
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "Unknown"

        # Get article date
        date_elem = soup.find("time") or soup.find(
            "meta", {"property": "article:published_time"}
        )
        date_text = date_elem.get("datetime", "") if date_elem else ""

        # Extract main content
        # Cloudflare blog uses article tag or specific content div
        content_div = (
            soup.find("article")
            or soup.find("div", class_="post-content")
            or soup.find("div", class_="markdown")
        )

        if not content_div:
            return incidents

        # Extract paragraphs with DDoS-related content
        paragraphs = content_div.find_all("p")

        for p in paragraphs:
            text = p.get_text(strip=True)

            # Filter for incident-like content
            if len(text) < 80:  # Too short
                continue

            # Look for keywords indicating attack description
            incident_indicators = [
                "attack",
                "ddos",
                "traffic",
                "mitigation",
                "volumetric",
                "amplification",
                "syn flood",
                "udp flood",
                "http flood",
                "botnet",
                "requests per second",
                "gbps",
                "tbps",
                "pps",
                "targeted",
                "overwhelmed",
                "disrupted",
                "outage",
            ]

            text_lower = text.lower()

            # Check if paragraph contains incident-related keywords
            if any(keyword in text_lower for keyword in incident_indicators):
                # Additional filtering - avoid generic/promotional content
                if not any(
                    skip in text_lower
                    for skip in ["cloudflare is", "our mission", "subscribe to"]
                ):
                    incidents.append(
                        {
                            "text": text,
                            "incident_type": "Denial of Service",
                            "source": "Cloudflare Blog",
                            "source_url": article_url,
                            "article_title": title_text,
                            "date": date_text,
                        }
                    )

    except requests.exceptions.RequestException as e:
        print(f"    Error fetching {article_url}: {e}")
    except Exception as e:
        print(f"    Parsing error for {article_url}: {e}")

    return incidents


def scrape_cloudflare_ddos(max_pages=5, max_articles=50):
    """
    Main scraping function

    Args:
        max_pages: Number of tag pages to browse
        max_articles: Maximum number of articles to process

    Returns:
        DataFrame with DDoS incidents
    """
    print("=" * 70)
    print("CLOUDFLARE DDOS BLOG SCRAPER")
    print("=" * 70)
    print()

    # Step 1: Get article URLs
    article_urls = get_article_urls(max_pages=max_pages)

    # Limit to max_articles
    article_urls = article_urls[:max_articles]

    print(f"Processing {len(article_urls)} articles...")
    print()

    # Step 2: Extract incidents from each article
    all_incidents = []

    for idx, url in enumerate(article_urls, 1):
        print(f"  [{idx}/{len(article_urls)}] {url.split('/')[-2][:40]}...", end=" ")

        incidents = extract_incident_narratives(url)
        all_incidents.extend(incidents)

        print(f"→ {len(incidents)} narratives")

        # Be respectful with delays
        if idx < len(article_urls):
            time.sleep(DELAY_SECONDS)

    # Step 3: Create DataFrame
    df = pd.DataFrame(all_incidents)

    if len(df) == 0:
        print("\n⚠️  No incidents extracted. The blog structure may have changed.")
        return df

    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=["text"])

    print()
    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Articles processed: {len(article_urls)}")
    print(f"Narratives extracted: {initial_len}")
    print(f"After deduplication: {len(df)}")
    print()

    # Show text length distribution
    if len(df) > 0:
        text_lengths = df["text"].str.len()
        print(f"Text length statistics:")
        print(f"  Mean: {text_lengths.mean():.0f} characters")
        print(f"  Median: {text_lengths.median():.0f} characters")
        print(f"  Min: {text_lengths.min():.0f} characters")
        print(f"  Max: {text_lengths.max():.0f} characters")

    return df


def save_dataset(df, output_path=OUTPUT_PATH):
    """
    Save extracted incidents to CSV

    Args:
        df: DataFrame with incidents
        output_path: Where to save
    """
    if len(df) == 0:
        print("\n⚠️  No data to save")
        return

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)

    print()
    print("=" * 70)
    print(f"✓ Saved {len(df)} incidents to: {output_path}")
    print("=" * 70)

    # Show samples
    print("\nSample incidents:")
    print("-" * 70)

    for idx, row in df.head(3).iterrows():
        print(f"\n{idx + 1}. From: {row['article_title'][:60]}...")
        print(f"   {row['text'][:200]}...")
        print()


def enhance_with_synthetic(df, target_count=500):
    """
    If we didn't get enough real incidents, supplement with enhanced synthetic

    Args:
        df: Existing incidents DataFrame
        target_count: Desired total incidents

    Returns:
        Enhanced DataFrame
    """
    current_count = len(df)

    if current_count >= target_count:
        print(f"\n✓ Have {current_count} incidents, no synthesis needed")
        return df

    needed = target_count - current_count
    print(f"\nGenerating {needed} synthetic incidents to reach {target_count}...")

    # Use patterns from real incidents to generate realistic ones
    templates = [
        "Network operations center detected traffic spike reaching {volume} Gbps targeting web infrastructure, identified as {attack_type} attack originating from {source}.",
        "Application layer attack observed with {rps} requests per second overwhelming {target}, legitimate users experienced {impact}.",
        "DNS infrastructure targeted by amplification attack generating {qps} queries per second using {vector}, causing {duration} service degradation.",
        "Volumetric attack detected against {asset} infrastructure, {protocol} flood consumed {bandwidth} of available capacity, mitigation activated.",
        "Multi-vector DDoS campaign identified targeting {services}, combining {vector1} and {vector2} attacks, peak traffic reached {volume}.",
        "Layer {layer} attack detected exploiting {vulnerability}, {count} malicious connections established, emergency traffic filtering engaged.",
        "Botnet-sourced attack overwhelmed edge routers with {pps} packets per second, {attack_type} flood blocked legitimate traffic for {duration}.",
        "Reflection attack leveraged {protocol} to amplify traffic by {factor}x, targeting {asset} with {volume} aggregate bandwidth.",
        "Application-specific attack targeted {endpoint} with {technique}, degraded response times to {latency}, affecting {users} concurrent users.",
    ]

    import random

    attack_types = [
        "volumetric DDoS",
        "SYN flood",
        "UDP amplification",
        "HTTP flood",
        "DNS flood",
        "NTP amplification",
    ]
    protocols = ["UDP", "TCP", "ICMP", "DNS", "NTP", "SSDP", "memcached"]
    sources = [
        "distributed botnet",
        "compromised IoT devices",
        "geographically dispersed sources",
        "cloud-based infrastructure",
    ]
    targets = [
        "customer-facing API",
        "authentication service",
        "content delivery network",
        "public DNS resolvers",
    ]
    services = ["web services", "API endpoints", "email infrastructure", "VoIP systems"]
    assets = ["edge", "core network", "customer", "public-facing"]
    endpoints = [
        "login endpoint",
        "/api/v1/auth",
        "checkout process",
        "search functionality",
    ]
    techniques = [
        "slowloris",
        "request flooding",
        "cache bypass",
        "resource exhaustion",
    ]

    synthetic_incidents = []

    for _ in range(needed):
        template = random.choice(templates)

        text = template.format(
            volume=f"{random.randint(50, 800)}",
            attack_type=random.choice(attack_types),
            source=random.choice(sources),
            rps=f"{random.randint(100, 500)}K",
            target=random.choice(targets),
            impact=random.choice(
                [
                    "severe degradation",
                    "complete service disruption",
                    "intermittent timeouts",
                    "elevated error rates",
                ]
            ),
            qps=f"{random.randint(10, 200)}M",
            vector=random.choice(protocols) + " reflection",
            duration=f"{random.randint(1, 8)} hours",
            asset=random.choice(assets),
            protocol=random.choice(protocols),
            bandwidth=f"{random.randint(60, 95)}%",
            services=random.choice(services),
            vector1=random.choice(attack_types),
            vector2=random.choice(attack_types),
            layer=random.choice(["3", "4", "7"]),
            vulnerability=random.choice(
                [
                    "keep-alive exhaustion",
                    "connection pool depletion",
                    "regex DoS",
                    "cache poisoning",
                ]
            ),
            count=f"{random.randint(10, 500)}K",
            pps=f"{random.randint(10, 200)}M",
            factor=random.randint(10, 100),
            endpoint=random.choice(endpoints),
            technique=random.choice(techniques),
            latency=f"{random.randint(5, 30)} seconds",
            users=f"{random.randint(1, 50)}K",
        )

        synthetic_incidents.append(
            {
                "text": text,
                "incident_type": "Denial of Service",
                "source": "Synthetic-Enhanced",
                "source_url": "",
                "article_title": "Generated",
                "date": "",
            }
        )

    synthetic_df = pd.DataFrame(synthetic_incidents)
    combined = pd.concat([df, synthetic_df], ignore_index=True)

    print(f"✓ Added {needed} synthetic incidents")
    print(f"  Total: {len(combined)} DDoS incidents")

    return combined


def main():
    """
    Main execution
    """
    # Scrape Cloudflare blog
    df = scrape_cloudflare_ddos(max_pages=3, max_articles=30)

    # Enhance with synthetic if needed
    df = enhance_with_synthetic(df, target_count=500)

    # Save
    save_dataset(df)

    print("\n✓ DDoS incident collection complete!")
    print("\nNext steps:")
    print("  1. Review the output file")
    print("  2. Merge with other datasets:")
    print("     python merge_all_datasets.py")
    print("  3. Retrain models")


if __name__ == "__main__":
    main()
