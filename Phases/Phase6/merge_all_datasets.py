"""
Enhanced Dataset Merger
Merges: CERT + Enron + VERIS + DDoS (Cloudflare/Multi-source)

This creates the final balanced dataset for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import resample

# Input paths - relative from Phases/Phase6/ directory
CERT_PATH = "data/cert_processed/cert_incidents.csv"
ENRON_PATH = "data/enron_processed/enron_incidents.csv"
VERIS_PATH = "data/veris_raw/veris_incidents.csv"
DDOS_MULTI_PATH = "data/scraped_incidents/cloudflare_ddos_incidents.csv"
DDOS_CLOUDFLARE_PATH = "data/scraped_incidents/ddos_incidents_enhanced.csv"

# Output path
OUTPUT_PATH = "data/real_incidents_balanced.csv"

# Target samples per class
TARGET_PER_CLASS = 3000


def load_dataset(path, dataset_name):
    """
    Load a single dataset with error handling

    Args:
        path: Path to CSV file
        dataset_name: Name for logging

    Returns:
        DataFrame or None
    """
    if Path(path).exists():
        print(f"Loading {dataset_name}...")
        df = pd.read_csv(path)
        print(f"  {dataset_name}: {len(df)} incidents")

        if "incident_type" in df.columns:
            print(f"  Distribution: {df['incident_type'].value_counts().to_dict()}")

        return df
    else:
        print(f"⚠️  {dataset_name} not found at {path}")
        return None


def load_all_datasets():
    """
    Load all available datasets

    Returns:
        List of DataFrames
    """
    print("=" * 70)
    print("LOADING ALL DATASETS")
    print("=" * 70)
    print()

    datasets = []

    # Load each dataset
    cert = load_dataset(CERT_PATH, "CERT")
    if cert is not None:
        datasets.append(cert)

    enron = load_dataset(ENRON_PATH, "Enron")
    if enron is not None:
        datasets.append(enron)

    veris = load_dataset(VERIS_PATH, "VERIS")
    if veris is not None:
        datasets.append(veris)

    # Try both DDoS sources (multi-source takes priority)
    ddos_multi = load_dataset(DDOS_MULTI_PATH, "DDoS (Multi-source)")
    if ddos_multi is not None:
        datasets.append(ddos_multi)
    else:
        # Fallback to Cloudflare-only
        ddos_cf = load_dataset(DDOS_CLOUDFLARE_PATH, "DDoS (Cloudflare)")
        if ddos_cf is not None:
            datasets.append(ddos_cf)

    if len(datasets) == 0:
        print("\n❌ No datasets found!")
        print("\nTo collect data, run:")
        print("  - python Phases/Phase2/cert_preprocess.py")
        print("  - python Phases/Phase2/enron_preprocess.py")
        print("  - python download_veris_incidents.py")
        print("  - python scrape_ddos_multi_source.py")
        return None

    print(f"\n✓ Loaded {len(datasets)} datasets")
    return datasets


def clean_and_normalize(df):
    """
    Clean and normalize incident data

    Args:
        df: DataFrame with text and incident_type columns

    Returns:
        Cleaned DataFrame
    """
    print("\n" + "=" * 70)
    print("CLEANING AND NORMALIZING")
    print("=" * 70)
    print()

    # Ensure required columns exist
    required_cols = {"text", "incident_type"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    initial_len = len(df)

    # Remove null values
    df = df.dropna(subset=["text", "incident_type"])
    print(f"Removed {initial_len - len(df)} rows with null values")

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])
    print(f"Removed {initial_len - len(df)} duplicate texts")

    # Remove very short texts
    df = df[df["text"].str.len() >= 50]
    print(f"Removed texts shorter than 50 characters")

    # Normalize incident type labels (standardize capitalization)
    df["incident_type"] = df["incident_type"].str.strip()

    # Standardize category names
    label_mapping = {
        "Denial of Service": "Denial of Service",
        "DoS": "Denial of Service",
        "DDoS": "Denial of Service",
        "Data Breach": "Data Breach",
        "Insider Misuse": "Insider Misuse",
        "Phishing": "Phishing",
        "Malware": "Malware",
        "Ransomware": "Ransomware",
    }

    df["incident_type"] = df["incident_type"].replace(label_mapping)

    print(f"\nFinal size after cleaning: {len(df)} incidents")
    print(f"\nClass distribution:")
    for label, count in df["incident_type"].value_counts().items():
        print(f"  {label:20s}: {count:4d}")

    return df


def balance_classes(df, target_per_class=TARGET_PER_CLASS):
    """
    Balance classes by resampling

    Args:
        df: DataFrame
        target_per_class: Target number of samples per class

    Returns:
        Balanced DataFrame
    """
    print("\n" + "=" * 70)
    print(f"BALANCING CLASSES (target: {target_per_class} per class)")
    print("=" * 70)
    print()

    balanced_dfs = []

    for label in sorted(df["incident_type"].unique()):
        subset = df[df["incident_type"] == label]
        current_count = len(subset)

        if current_count > target_per_class:
            # Downsample
            print(
                f"{label:20s}: {current_count:4d} → {target_per_class:4d} (downsampling)"
            )
            subset = resample(
                subset, n_samples=target_per_class, random_state=42, replace=False
            )
        elif current_count < target_per_class:
            # Upsample (with replacement)
            print(
                f"{label:20s}: {current_count:4d} → {target_per_class:4d} (upsampling)"
            )
            subset = resample(
                subset, n_samples=target_per_class, random_state=42, replace=True
            )
        else:
            print(f"{label:20s}: {current_count:4d} (no change needed)")

        balanced_dfs.append(subset)

    # Combine and shuffle
    balanced = pd.concat(balanced_dfs, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nTotal incidents after balancing: {len(balanced)}")

    return balanced


def analyze_dataset(df):
    """
    Print comprehensive dataset statistics

    Args:
        df: DataFrame to analyze
    """
    print("\n" + "=" * 70)
    print("FINAL DATASET ANALYSIS")
    print("=" * 70)

    print(f"\nTotal incidents: {len(df)}")

    # Class distribution
    print("\nClass distribution:")
    counts = df["incident_type"].value_counts()
    for label, count in counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label:20s}: {count:4d} ({percentage:5.1f}%)")

    # Text length statistics
    print("\nText length statistics:")
    text_lengths = df["text"].str.len()
    print(f"  Mean:   {text_lengths.mean():6.0f} characters")
    print(f"  Median: {text_lengths.median():6.0f} characters")
    print(f"  Min:    {text_lengths.min():6.0f} characters")
    print(f"  Max:    {text_lengths.max():6.0f} characters")
    print(f"  Std:    {text_lengths.std():6.0f} characters")

    # Source distribution
    if "source" in df.columns:
        print("\nSource distribution:")
        source_counts = df["source"].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {source:25s}: {count:4d} ({percentage:5.1f}%)")

    # Sample incidents
    print("\n" + "=" * 70)
    print("SAMPLE INCIDENTS (one per class)")
    print("=" * 70)

    for label in sorted(df["incident_type"].unique()):
        samples = df[df["incident_type"] == label]
        if len(samples) > 0:
            sample = samples.iloc[0]
            print(f"\n{label}:")
            print(f"  Source: {sample.get('source', 'Unknown')}")
            print(f"  Text ({len(sample['text'])} chars):")
            print(f"  {sample['text'][:300]}...")


def save_dataset(df, output_path=OUTPUT_PATH):
    """
    Save final dataset

    Args:
        df: DataFrame to save
        output_path: Where to save
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Keep only essential columns
    essential_cols = ["text", "incident_type"]
    if "source" in df.columns:
        essential_cols.append("source")

    df_to_save = df[essential_cols].copy()

    # Save
    df_to_save.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print(f"✓ Dataset saved to: {output_path}")
    print("=" * 70)

    print("\nNext steps:")
    print("  1. Review the dataset:")
    print(f"     head {output_path}")
    print("  2. Retrain models:")
    print("     python Phases/Phase5/train_and_evaluate_pipeline.py")
    print("  3. Evaluate improvement:")
    print("     python Phases/Phase5/generate_plots.py")


def create_data_summary(df):
    """
    Create a summary report of the dataset
    """
    summary_path = OUTPUT_PATH.replace(".csv", "_summary.txt")

    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DATASET SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Generated: {pd.Timestamp.now()}\n\n")

        f.write(f"Total incidents: {len(df)}\n")
        f.write(f"Number of classes: {df['incident_type'].nunique()}\n\n")

        f.write("Class Distribution:\n")
        for label, count in df["incident_type"].value_counts().items():
            pct = (count / len(df)) * 100
            f.write(f"  {label:20s}: {count:4d} ({pct:5.1f}%)\n")

        f.write("\nText Statistics:\n")
        text_lengths = df["text"].str.len()
        f.write(f"  Mean length:   {text_lengths.mean():6.0f} chars\n")
        f.write(f"  Median length: {text_lengths.median():6.0f} chars\n")
        f.write(f"  Min length:    {text_lengths.min():6.0f} chars\n")
        f.write(f"  Max length:    {text_lengths.max():6.0f} chars\n")

        if "source" in df.columns:
            f.write("\nData Sources:\n")
            for source, count in df["source"].value_counts().items():
                pct = (count / len(df)) * 100
                f.write(f"  {source:25s}: {count:4d} ({pct:5.1f}%)\n")

    print(f"\n✓ Summary report saved to: {summary_path}")


def main():
    """
    Main merging workflow
    """
    print("\n" + "=" * 70)
    print("ENHANCED DATASET MERGER")
    print("CERT + Enron + VERIS + DDoS")
    print("=" * 70)
    print()

    # Load all datasets
    datasets = load_all_datasets()

    if datasets is None or len(datasets) == 0:
        print("\n❌ Cannot proceed without data")
        return

    # Merge all datasets
    print("\n" + "=" * 70)
    print("MERGING DATASETS")
    print("=" * 70)
    print()

    merged = pd.concat(datasets, ignore_index=True)
    print(f"✓ Merged {len(datasets)} datasets → {len(merged)} total incidents")

    # Clean and normalize
    cleaned = clean_and_normalize(merged)

    # Balance classes
    balanced = balance_classes(cleaned, target_per_class=TARGET_PER_CLASS)

    # Analyze final dataset
    analyze_dataset(balanced)

    # Save
    save_dataset(balanced)

    # Create summary report
    create_data_summary(balanced)

    print("\n" + "=" * 70)
    print("✓ DATASET PREPARATION COMPLETE!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
