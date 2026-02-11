# """
# Alternative DDoS incident scraper using RSS feeds and multiple sources

# This approach is more reliable than HTML scraping because:
# - RSS feeds are more stable
# - Multiple sources = more data
# - Structured XML parsing
# """

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time
# from datetime import datetime
# import xml.etree.ElementTree as ET
# from pathlib import Path
# import re

# OUTPUT_PATH = "data/scraped_incidents/ddos_incidents_multi_source.csv"
# DELAY_SECONDS = 2


# def scrape_cloudflare_rss():
#     """
#     Scrape Cloudflare blog RSS feed for DDoS articles
#     More reliable than HTML scraping
#     """
#     incidents = []

#     print("Fetching Cloudflare DDoS articles via RSS...")

#     # Cloudflare blog RSS feed
#     rss_url = "https://blog.cloudflare.com/rss/"

#     try:
#         response = requests.get(rss_url, timeout=15)
#         response.raise_for_status()

#         # Parse RSS XML
#         root = ET.fromstring(response.content)

#         # Find all items (articles)
#         for item in root.findall(".//item"):
#             title = item.find("title").text if item.find("title") is not None else ""
#             link = item.find("link").text if item.find("link") is not None else ""
#             description = (
#                 item.find("description").text
#                 if item.find("description") is not None
#                 else ""
#             )
#             pub_date = (
#                 item.find("pubDate").text if item.find("pubDate") is not None else ""
#             )

#             # Check if article is about DDoS
#             title_lower = title.lower()
#             desc_lower = description.lower() if description else ""

#             if any(
#                 keyword in title_lower or keyword in desc_lower
#                 for keyword in ["ddos", "denial of service", "attack", "mitigation"]
#             ):
#                 # Clean HTML from description
#                 if description:
#                     clean_desc = BeautifulSoup(description, "html.parser").get_text()

#                     if len(clean_desc) > 100:
#                         incidents.append(
#                             {
#                                 "text": clean_desc,
#                                 "incident_type": "Denial of Service",
#                                 "source": "Cloudflare Blog RSS",
#                                 "article_title": title,
#                                 "source_url": link,
#                                 "date": pub_date,
#                             }
#                         )

#         print(f"  ✓ Found {len(incidents)} DDoS-related articles")

#     except Exception as e:
#         print(f"  Error fetching RSS: {e}")

#     return incidents


# def scrape_akamai_threats():
#     """
#     Scrape Akamai threat research for DDoS incidents
#     Akamai publishes excellent DDoS attack reports
#     """
#     incidents = []

#     print("\nFetching Akamai threat reports...")

#     # Akamai threat research blog
#     url = "https://www.akamai.com/blog/security"

#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
#     }

#     try:
#         response = requests.get(url, headers=headers, timeout=15)
#         response.raise_for_status()

#         soup = BeautifulSoup(response.content, "html.parser")

#         # Find article previews
#         articles = soup.find_all(
#             ["article", "div"], class_=re.compile("post|article|card")
#         )

#         for article in articles[:20]:  # Process first 20
#             title_elem = article.find(["h2", "h3", "h4"])
#             if not title_elem:
#                 continue

#             title = title_elem.get_text(strip=True)

#             # Check if DDoS-related
#             if any(
#                 keyword in title.lower()
#                 for keyword in ["ddos", "denial", "attack", "botnet"]
#             ):
#                 # Get description/excerpt
#                 desc_elem = article.find("p")
#                 if desc_elem:
#                     text = desc_elem.get_text(strip=True)

#                     if len(text) > 80:
#                         # Get link
#                         link_elem = article.find("a", href=True)
#                         link = link_elem["href"] if link_elem else ""
#                         if link and not link.startswith("http"):
#                             link = "https://www.akamai.com" + link

#                         incidents.append(
#                             {
#                                 "text": text,
#                                 "incident_type": "Denial of Service",
#                                 "source": "Akamai Security Blog",
#                                 "article_title": title,
#                                 "source_url": link,
#                                 "date": "",
#                             }
#                         )

#         print(f"  ✓ Found {len(incidents)} DDoS incidents")

#     except Exception as e:
#         print(f"  Error: {e}")

#     time.sleep(DELAY_SECONDS)
#     return incidents


# def scrape_arbor_networks():
#     """
#     Scrape Arbor Networks (NETSCOUT) DDoS reports
#     Industry leader in DDoS protection - excellent data
#     """
#     incidents = []

#     print("\nFetching NETSCOUT/Arbor DDoS reports...")

#     url = "https://www.netscout.com/blog/asert"

#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
#     }

#     try:
#         response = requests.get(url, headers=headers, timeout=15)
#         response.raise_for_status()

#         soup = BeautifulSoup(response.content, "html.parser")

#         # Find blog articles
#         articles = soup.find_all(
#             ["article", "div"], class_=re.compile("post|blog|article")
#         )

#         for article in articles[:20]:
#             title_elem = article.find(["h2", "h3", "a"])
#             if not title_elem:
#                 continue

#             title = title_elem.get_text(strip=True)

#             if any(
#                 keyword in title.lower()
#                 for keyword in ["ddos", "attack", "amplification", "reflection"]
#             ):
#                 desc_elem = article.find("p")
#                 if desc_elem:
#                     text = desc_elem.get_text(strip=True)

#                     if len(text) > 80:
#                         link_elem = article.find("a", href=True)
#                         link = link_elem["href"] if link_elem else ""
#                         if link and not link.startswith("http"):
#                             link = "https://www.netscout.com" + link

#                         incidents.append(
#                             {
#                                 "text": text,
#                                 "incident_type": "Denial of Service",
#                                 "source": "NETSCOUT ASERT Blog",
#                                 "article_title": title,
#                                 "source_url": link,
#                                 "date": "",
#                             }
#                         )

#         print(f"  ✓ Found {len(incidents)} DDoS incidents")

#     except Exception as e:
#         print(f"  Error: {e}")

#     time.sleep(DELAY_SECONDS)
#     return incidents


# def generate_realistic_ddos_incidents(count=300):
#     """
#     Generate realistic DDoS incidents based on actual attack patterns

#     Uses real attack characteristics from industry reports
#     """
#     print(f"\nGenerating {count} realistic DDoS incidents...")

#     # Real attack patterns from NETSCOUT/Cloudflare reports
#     attack_templates = [
#         # Volumetric attacks
#         "Network monitoring detected {volume} Gbps volumetric attack targeting {target}, {protocol} flood originated from {source}, lasting {duration}.",
#         # Amplification attacks
#         "{protocol} amplification attack leveraged {count} reflectors to amplify traffic by {factor}x, peak bandwidth reached {volume} Gbps against {asset}.",
#         # Application layer
#         "Application layer attack overwhelmed {service} with {rps} requests per second, {technique} exploitation caused {impact}, mitigation engaged after {delay}.",
#         # Multi-vector
#         "Multi-vector DDoS campaign combined {vector1}, {vector2}, and {vector3}, targeting {target} infrastructure, total attack volume {volume} Gbps.",
#         # Protocol attacks
#         "{protocol} flood attack consumed server resources with {pps} packets per second, connection table exhaustion affected {count} legitimate users.",
#         # Botnet attacks
#         "Botnet comprising {botnet_size} compromised devices launched {attack_type} attack, distributed sources across {countries} countries, sustained for {duration}.",
#         # Reflection attacks
#         "DNS reflection attack using {count} open resolvers generated {qps} million queries per second, targeting authoritative nameservers of {target}.",
#         # Layer 7 attacks
#         "Layer 7 attack exploited {vulnerability} in {service}, slowloris technique maintained {connections} concurrent connections, degraded response time to {latency}.",
#         # SYN floods
#         "SYN flood attack initiated {pps} million packets per second against edge infrastructure, half-open connections depleted {resource}, {duration} service disruption.",
#         # UDP floods
#         "UDP flood targeted {port} with randomized payloads at {pps} million PPS, saturated {bandwidth} of available bandwidth, forced traffic blackholing.",
#         # Memcached amplification
#         "Memcached amplification attack achieved {factor}x amplification factor using {count} exposed servers, peak attack size {volume} Tbps, unprecedented scale.",
#         # NTP amplification
#         "NTP amplification leveraged monlist command on {count} vulnerable servers, generated {volume} Gbps attacking {target}, {duration} outage resulted.",
#     ]

#     # Real-world attack characteristics
#     import random

#     protocols = [
#         "UDP",
#         "TCP",
#         "ICMP",
#         "DNS",
#         "NTP",
#         "SSDP",
#         "memcached",
#         "CLDAP",
#         "LDAP",
#     ]
#     targets = [
#         "customer API gateway",
#         "public DNS infrastructure",
#         "web application cluster",
#         "edge routers",
#         "content delivery network",
#         "authentication service",
#     ]
#     assets = [
#         "public-facing web servers",
#         "DNS resolvers",
#         "API endpoints",
#         "edge network",
#         "load balancers",
#     ]
#     sources = [
#         "Mirai botnet variant",
#         "geographically distributed botnet",
#         "compromised IoT devices",
#         "cloud-based attack infrastructure",
#         "residential proxy network",
#     ]
#     services = [
#         "authentication API",
#         "content management system",
#         "e-commerce platform",
#         "customer portal",
#         "payment processing gateway",
#     ]
#     techniques = [
#         "HTTP POST flood",
#         "Slowloris",
#         "R-U-Dead-Yet",
#         "cache bypass",
#         "WordPress pingback",
#         "XML-RPC amplification",
#     ]
#     vulnerabilities = [
#         "keep-alive abuse",
#         "connection pooling exhaustion",
#         "SSL/TLS handshake flood",
#         "regex DoS",
#         "algorithmic complexity",
#     ]
#     resources = [
#         "connection tracking table",
#         "NAT translation entries",
#         "session state memory",
#         "CPU cycles",
#         "bandwidth capacity",
#     ]

#     incidents = []

#     for _ in range(count):
#         template = random.choice(attack_templates)

#         incident_text = template.format(
#             volume=random.choice(
#                 [
#                     random.randint(50, 400),
#                     random.randint(400, 800),
#                     random.randint(1, 3) * 1000,
#                 ]
#             ),
#             target=random.choice(targets),
#             protocol=random.choice(protocols),
#             source=random.choice(sources),
#             duration=random.choice(
#                 [
#                     f"{random.randint(1, 4)} hours",
#                     f"{random.randint(30, 180)} minutes",
#                     f"{random.randint(5, 12)} hours",
#                 ]
#             ),
#             count=f"{random.randint(10, 500)}K"
#             if random.random() > 0.3
#             else f"{random.randint(1, 50)}M",
#             factor=random.randint(10, 51000),
#             asset=random.choice(assets),
#             service=random.choice(services),
#             rps=f"{random.randint(100, 900)}K"
#             if random.random() > 0.5
#             else f"{random.randint(1, 20)}M",
#             technique=random.choice(techniques),
#             impact=random.choice(
#                 [
#                     "complete service outage",
#                     "severe performance degradation",
#                     "500 errors for legitimate traffic",
#                     "timeouts across all endpoints",
#                 ]
#             ),
#             delay=f"{random.randint(2, 15)} minutes",
#             vector1=random.choice(protocols) + " flood",
#             vector2=random.choice(techniques),
#             vector3=random.choice(["SYN flood", "ACK flood", "RST flood", "FIN flood"]),
#             pps=f"{random.randint(10, 500)}M",
#             botnet_size=f"{random.randint(50, 500)}K"
#             if random.random() > 0.3
#             else f"{random.randint(1, 10)}M",
#             attack_type=random.choice(
#                 ["volumetric", "application-layer", "protocol", "hybrid"]
#             ),
#             countries=random.randint(50, 180),
#             qps=random.randint(10, 200),
#             vulnerability=random.choice(vulnerabilities),
#             connections=f"{random.randint(10, 500)}K",
#             latency=f"{random.randint(10, 60)} seconds",
#             resource=random.choice(resources),
#             bandwidth=f"{random.randint(70, 100)}%",
#             port=random.choice(
#                 ["53/UDP", "80/TCP", "443/TCP", "123/UDP", "1900/UDP", "11211/UDP"]
#             ),
#         )

#         incidents.append(
#             {
#                 "text": incident_text,
#                 "incident_type": "Denial of Service",
#                 "source": "Synthetic-Realistic",
#                 "article_title": "Generated from industry patterns",
#                 "source_url": "",
#                 "date": "",
#             }
#         )

#     print(f"  ✓ Generated {len(incidents)} incidents")
#     return incidents


# def main():
#     """
#     Collect DDoS incidents from multiple sources
#     """
#     print("=" * 70)
#     print("MULTI-SOURCE DDOS INCIDENT COLLECTOR")
#     print("=" * 70)
#     print()

#     all_incidents = []

#     # Method 1: Cloudflare RSS (most reliable)
#     all_incidents.extend(scrape_cloudflare_rss())
#     time.sleep(DELAY_SECONDS)

#     # Method 2: Akamai blog
#     all_incidents.extend(scrape_akamai_threats())

#     # Method 3: NETSCOUT/Arbor
#     all_incidents.extend(scrape_arbor_networks())

#     # Method 4: Generate realistic synthetic to supplement
#     scraped_count = len(all_incidents)
#     print(f"\nTotal scraped: {scraped_count} incidents")

#     if scraped_count < 400:
#         needed = 400 - scraped_count
#         all_incidents.extend(generate_realistic_ddos_incidents(needed))

#     # Create DataFrame
#     df = pd.DataFrame(all_incidents)

#     # Remove duplicates
#     initial_len = len(df)
#     df = df.drop_duplicates(subset=["text"])

#     print()
#     print("=" * 70)
#     print("COLLECTION SUMMARY")
#     print("=" * 70)
#     print(f"Total incidents: {initial_len}")
#     print(f"After deduplication: {len(df)}")
#     print()
#     print("Source breakdown:")
#     print(df["source"].value_counts())

#     # Save
#     Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(OUTPUT_PATH, index=False)

#     print()
#     print(f"✓ Saved to: {OUTPUT_PATH}")

#     # Show samples
#     print()
#     print("=" * 70)
#     print("SAMPLE INCIDENTS")
#     print("=" * 70)

#     for idx, row in df.head(5).iterrows():
#         print(f"\n{idx + 1}. [{row['source']}]")
#         print(f"   {row['text'][:250]}...")

#     print()
#     print("=" * 70)
#     print("✓ DDoS incident collection complete!")
#     print("=" * 70)
#     print()
#     print("Next steps:")
#     print("  1. Review the output file")
#     print("  2. Merge with other incident types:")
#     print("     python merge_all_datasets.py")


# if __name__ == "__main__":
#     main()


"""
Enhanced DDoS Incident Scraper
- Better HTML parsing
- More sources
- PDF report extraction
- Fallback to LLM-based realistic generation (only if needed)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from pathlib import Path
import re
from datetime import datetime

OUTPUT_PATH = "data/scraped_incidents/ddos_incidents_enhanced.csv"
DELAY_SECONDS = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def scrape_cloudflare_blog_direct(max_articles=50):
    """
    Scrape Cloudflare blog directly with better parsing
    """
    incidents = []

    print("Scraping Cloudflare DDoS blog posts...")

    # Try multiple tag/category pages
    urls = [
        "https://blog.cloudflare.com/tag/ddos/",
        "https://blog.cloudflare.com/tag/attacks/",
        "https://blog.cloudflare.com/tag/security/",
    ]

    for base_url in urls:
        try:
            for page in range(1, 6):  # Try 5 pages
                if page == 1:
                    url = base_url
                else:
                    url = f"{base_url}page/{page}/"

                print(f"  Fetching {url}...")
                response = requests.get(url, headers=HEADERS, timeout=15)

                if response.status_code != 200:
                    break

                soup = BeautifulSoup(response.content, "html.parser")

                # Find all article cards/links
                articles = soup.find_all(
                    ["article", "div"], class_=re.compile("post|card|article")
                )

                if not articles:
                    # Try alternative selectors
                    articles = soup.find_all("a", href=re.compile("/[0-9]{4}/"))

                for article in articles:
                    try:
                        # Get article URL
                        link = article.find("a", href=True)
                        if not link:
                            link = article if article.name == "a" else None

                        if not link:
                            continue

                        article_url = link["href"]
                        if not article_url.startswith("http"):
                            article_url = "https://blog.cloudflare.com" + article_url

                        # Check if DDoS-related
                        if (
                            "ddos" not in article_url.lower()
                            and "attack" not in article_url.lower()
                        ):
                            continue

                        print(f"    Found: {article_url}")

                        # Fetch full article
                        time.sleep(DELAY_SECONDS)
                        article_response = requests.get(
                            article_url, headers=HEADERS, timeout=15
                        )
                        article_soup = BeautifulSoup(
                            article_response.content, "html.parser"
                        )

                        # Extract all paragraphs
                        paragraphs = article_soup.find_all("p")

                        for p in paragraphs:
                            text = p.get_text(strip=True)

                            if len(text) < 100:
                                continue

                            # Check for DDoS-related keywords
                            keywords = [
                                "attack",
                                "ddos",
                                "gbps",
                                "requests per second",
                                "botnet",
                                "volumetric",
                                "amplification",
                                "flood",
                                "mitigation",
                                "traffic",
                            ]

                            if any(kw in text.lower() for kw in keywords):
                                incidents.append(
                                    {
                                        "text": text,
                                        "incident_type": "Denial of Service",
                                        "source": "Cloudflare Blog",
                                        "source_url": article_url,
                                        "date": datetime.now().strftime("%Y-%m-%d"),
                                    }
                                )

                    except Exception as e:
                        continue

                time.sleep(DELAY_SECONDS)

                if len(incidents) >= max_articles:
                    break

            if len(incidents) >= max_articles:
                break

        except Exception as e:
            print(f"  Error on {base_url}: {e}")
            continue

    print(f"  ✓ Extracted {len(incidents)} paragraphs from Cloudflare")
    return incidents


def scrape_arbor_reports():
    """
    Scrape NETSCOUT Arbor DDoS reports
    """
    incidents = []

    print("\nScraping NETSCOUT Arbor reports...")

    urls = [
        "https://www.netscout.com/blog",
        "https://www.netscout.com/threatreport",
    ]

    for url in urls:
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all text content
            main_content = soup.find("main") or soup.find("article") or soup
            paragraphs = main_content.find_all("p")

            for p in paragraphs:
                text = p.get_text(strip=True)

                if len(text) < 100:
                    continue

                if any(
                    kw in text.lower() for kw in ["ddos", "attack", "gbps", "botnet"]
                ):
                    incidents.append(
                        {
                            "text": text,
                            "incident_type": "Denial of Service",
                            "source": "NETSCOUT",
                            "source_url": url,
                            "date": "",
                        }
                    )

            time.sleep(DELAY_SECONDS)

        except Exception as e:
            print(f"  Error: {e}")

    print(f"  ✓ Extracted {len(incidents)} paragraphs from NETSCOUT")
    return incidents


def use_llm_to_generate_from_real_patterns(real_examples, target_count=500):
    """
    Use Groq LLM to generate realistic incidents based on real examples
    Only use if we have some real examples to learn from
    """
    print(
        f"\nUsing LLM to generate {target_count} realistic incidents from real patterns..."
    )

    try:
        from groq import Groq
        from dotenv import load_dotenv

        load_dotenv()
        client = Groq()

        # Take best real examples
        sample_text = "\n\n".join([ex["text"][:500] for ex in real_examples[:5]])

        generated = []

        # Generate in batches
        batch_size = 10
        batches = target_count // batch_size

        for batch in range(batches):
            prompt = f"""You are generating realistic DDoS incident descriptions for a security dataset.

Based on these REAL examples from industry reports:

{sample_text}

Generate {batch_size} NEW, unique DDoS incident descriptions. Each should:
- Be 150-300 words
- Include specific metrics (Gbps, RPS, PPS)
- Mention attack vectors (UDP flood, SYN flood, amplification, etc.)
- Describe detection and impact
- Use professional security analyst language
- Be DIFFERENT from the examples

Format: Return ONLY the incident descriptions, separated by "---INCIDENT---"

Generate {batch_size} incidents now:"""

            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",  # Better model for quality
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,  # High creativity
                    max_tokens=4000,
                )

                content = response.choices[0].message.content

                # Split by separator
                descriptions = content.split("---INCIDENT---")

                for desc in descriptions:
                    desc = desc.strip()
                    if len(desc) > 100:
                        generated.append(
                            {
                                "text": desc,
                                "incident_type": "Denial of Service",
                                "source": "LLM-Enhanced (from real patterns)",
                                "source_url": "",
                                "date": "",
                            }
                        )

                print(f"  Generated batch {batch + 1}/{batches}")
                time.sleep(1)  # API rate limit

            except Exception as e:
                print(f"  Error in batch {batch}: {e}")
                continue

        print(f"  ✓ Generated {len(generated)} LLM-based incidents")
        return generated

    except Exception as e:
        print(f"  ✗ LLM generation failed: {e}")
        return []


def main():
    print("=" * 70)
    print("ENHANCED DDOS INCIDENT COLLECTOR")
    print("Goal: 3000 high-quality incidents")
    print("=" * 70)
    print()

    all_incidents = []

    # Phase 1: Intensive web scraping
    print("PHASE 1: Web Scraping")
    print("-" * 70)

    all_incidents.extend(scrape_cloudflare_blog_direct(max_articles=200))
    all_incidents.extend(scrape_arbor_reports())

    # Remove duplicates
    df = pd.DataFrame(all_incidents)
    df = df.drop_duplicates(subset=["text"])

    real_count = len(df)
    print(f"\n✓ Total real incidents scraped: {real_count}")

    # Phase 2: LLM generation if needed
    if real_count < 3000:
        needed = 3000 - real_count
        print(f"\nPHASE 2: LLM-Enhanced Generation")
        print("-" * 70)
        print(f"Need {needed} more incidents to reach 3000")

        if real_count > 0:
            # Use real examples as templates
            llm_incidents = use_llm_to_generate_from_real_patterns(
                all_incidents[:20],  # Use best 20 examples
                target_count=needed,
            )

            if llm_incidents:
                all_incidents.extend(llm_incidents)
                df = pd.DataFrame(all_incidents)
                df = df.drop_duplicates(subset=["text"])

    # Final statistics
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\nTotal incidents: {len(df)}")
    print("\nSource breakdown:")
    print(df["source"].value_counts())

    # Text quality check
    print("\nQuality metrics:")
    lengths = df["text"].str.len()
    print(f"  Mean length: {lengths.mean():.0f} chars")
    print(f"  Median length: {lengths.median():.0f} chars")

    # Save
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✓ Saved to: {OUTPUT_PATH}")

    # Show samples
    print("\n" + "=" * 70)
    print("SAMPLE INCIDENTS")
    print("=" * 70)

    for idx, row in df.head(3).iterrows():
        print(f"\n{idx + 1}. [{row['source']}]")
        print(f"   {row['text'][:300]}...")

    return df


if __name__ == "__main__":
    main()
