"""
Extract DDoS incidents from VERIS with enhanced filtering

VERIS has ~200-300 DDoS incidents. We'll extract all of them
and then supplement with LLM-based generation using those as templates.
"""

import json
import pandas as pd
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import time

load_dotenv()
client = Groq()


def extract_ddos_from_veris(veris_dir="VCDB/data/json/validated"):
    """
    Extract ALL DDoS/DoS incidents from VERIS with detailed text
    """
    print("Extracting DDoS incidents from VERIS...")

    veris_path = Path(veris_dir)

    if not veris_path.exists():
        print(f"Error: VERIS not found at {veris_path}")
        return []

    incidents = []
    json_files = list(veris_path.glob("*.json"))

    print(f"Processing {len(json_files)} VERIS incident files...")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                incident = json.load(f)

            # Check if it's a DoS/DDoS incident
            action = incident.get("action", {})
            hacking = action.get("hacking", {})

            # DoS indicators
            is_dos = False

            # Direct DoS variety
            if "DoS" in hacking.get("variety", []) or "DDoS" in hacking.get(
                "variety", []
            ):
                is_dos = True

            # Check action varieties for denial
            for action_type in action.values():
                if isinstance(action_type, dict):
                    varieties = action_type.get("variety", [])
                    if any(
                        "DoS" in v or "DDoS" in v or "Denial" in v for v in varieties
                    ):
                        is_dos = True

            # Check summary for DoS keywords
            summary = incident.get("summary", "")
            if any(
                kw in summary.lower()
                for kw in [
                    "ddos",
                    "dos attack",
                    "denial of service",
                    "flood",
                    "volumetric",
                ]
            ):
                is_dos = True

            if not is_dos:
                continue

            # Extract rich text description
            text_parts = []

            # Summary
            if summary:
                text_parts.append(summary)

            # Notes
            notes = incident.get("notes", "")
            if notes:
                text_parts.append(notes)

            # Victim description
            victim = incident.get("victim", {})
            if victim.get("notes"):
                text_parts.append(victim["notes"])

            # Action description
            if action.get("notes"):
                text_parts.append(action["notes"])

            # Combine all text
            full_text = " ".join(text_parts)

            if len(full_text) < 50:
                continue

            incidents.append(
                {
                    "text": full_text,
                    "incident_type": "Denial of Service",
                    "source": "VERIS",
                    "source_url": "",
                    "date": incident.get("timeline", {})
                    .get("incident", {})
                    .get("year", ""),
                }
            )

        except Exception as e:
            continue

    print(f"✓ Extracted {len(incidents)} DDoS incidents from VERIS")
    return incidents


def generate_from_veris_templates(veris_incidents, target_total=3000):
    """
    Use VERIS incidents as high-quality templates for LLM generation
    """
    current_count = len(veris_incidents)
    needed = target_total - current_count

    if needed <= 0:
        print(f"Already have {current_count} incidents, no generation needed")
        return []

    print(f"\nGenerating {needed} additional incidents using VERIS templates...")

    generated = []

    # Select best examples (longest, most detailed)
    veris_df = pd.DataFrame(veris_incidents)
    veris_df["length"] = veris_df["text"].str.len()
    best_examples = veris_df.nlargest(10, "length")

    # Create example pool
    examples_text = "\n\n---\n\n".join(
        [
            f"Example {i + 1}:\n{row['text'][:500]}"
            for i, (_, row) in enumerate(best_examples.iterrows())
        ]
    )

    # Generate in batches of 20
    batch_size = 20
    num_batches = (needed + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        batch_count = min(batch_size, needed - len(generated))

        prompt = f"""You are a cybersecurity expert creating realistic DDoS/DoS incident descriptions for training a machine learning model.

Here are REAL DDoS incidents from the VERIS database:

{examples_text}

Generate {batch_count} NEW, unique DDoS incident descriptions that:
1. Match the style and detail level of the examples
2. Include specific technical details (attack vectors, traffic volumes, duration)
3. Describe detection method and business impact
4. Are 150-400 words each
5. Use varied attack types: volumetric, application-layer, protocol attacks, amplification
6. Include realistic metrics (Gbps, RPS, PPS, botnet sizes)
7. Are completely DIFFERENT from each other and from the examples

Return ONLY the incident descriptions, separated by:
###INCIDENT###

Generate {batch_count} incidents now:"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=8000,
            )

            content = response.choices[0].message.content

            # Split incidents
            descriptions = [
                d.strip() for d in content.split("###INCIDENT###") if d.strip()
            ]

            for desc in descriptions:
                if len(desc) > 100:
                    generated.append(
                        {
                            "text": desc,
                            "incident_type": "Denial of Service",
                            "source": "LLM-Generated (VERIS templates)",
                            "source_url": "",
                            "date": "",
                        }
                    )

            print(
                f"  Batch {batch_num + 1}/{num_batches}: Generated {len(descriptions)} incidents"
            )
            time.sleep(2)  # Rate limit

        except Exception as e:
            print(f"  Error in batch {batch_num + 1}: {e}")
            continue

    print(f"✓ Generated {len(generated)} LLM-based incidents")
    return generated


def main():
    print("=" * 70)
    print("DDOS DATA COLLECTION - VERIS + LLM APPROACH")
    print("Target: 3000 incidents")
    print("=" * 70)
    print()

    # Step 1: Extract from VERIS
    veris_incidents = extract_ddos_from_veris()

    if len(veris_incidents) == 0:
        print("\n❌ No VERIS incidents found. Make sure VCDB is cloned.")
        print("Run: git clone https://github.com/vz-risk/VCDB.git")
        return

    # Step 2: Generate additional using LLM
    generated_incidents = generate_from_veris_templates(
        veris_incidents, target_total=3000
    )

    # Combine
    all_incidents = veris_incidents + generated_incidents

    # Create DataFrame
    df = pd.DataFrame(all_incidents)
    df = df.drop_duplicates(subset=["text"])

    # Statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"\nTotal incidents: {len(df)}")
    print(f"  Real (VERIS): {len(veris_incidents)}")
    print(f"  LLM-Generated: {len(generated_incidents)}")

    print("\nSource breakdown:")
    print(df["source"].value_counts())

    print("\nQuality metrics:")
    lengths = df["text"].str.len()
    print(f"  Mean length: {lengths.mean():.0f} characters")
    print(f"  Median length: {lengths.median():.0f} characters")
    print(f"  Min length: {lengths.min():.0f} characters")
    print(f"  Max length: {lengths.max():.0f} characters")

    # Save
    output_path = "../../data/scraped_incidents/ddos_incidents_veris_llm.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved to: {output_path}")

    # Show samples
    print("\n" + "=" * 70)
    print("SAMPLE INCIDENTS")
    print("=" * 70)

    # Show VERIS examples
    veris_samples = df[df["source"] == "VERIS"].head(2)
    print("\nREAL VERIS incidents:")
    for idx, row in veris_samples.iterrows():
        print(f"\n{idx + 1}. {row['text'][:400]}...")

    # Show LLM examples
    if len(generated_incidents) > 0:
        llm_samples = df[df["source"] == "LLM-Generated (VERIS templates)"].head(2)
        print("\n\nLLM-GENERATED incidents:")
        for idx, row in llm_samples.iterrows():
            print(f"\n{idx + 1}. {row['text'][:400]}...")


if __name__ == "__main__":
    main()
