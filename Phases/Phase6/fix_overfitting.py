import pandas as pd
import numpy as np
from pathlib import Path


def check_current_dataset():
    """
    Check what's in current dataset
    """
    print("=" * 70)
    print("CHECKING CURRENT DATASET")
    print("=" * 70)
    print()

    csv_path = "data/real_incidents_balanced.csv"

    if not Path(csv_path).exists():
        print(f"❌ Dataset not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    print(f"Total incidents: {len(df)}")
    print()

    print("Class distribution:")
    print(df["incident_type"].value_counts())
    print()

    if "source" in df.columns:
        print("Source breakdown by class:")
        print(df.groupby(["incident_type", "source"]).size().unstack(fill_value=0))
        print()

        # Check synthetic percentage
        for incident_type in ["Denial of Service", "Malware", "Ransomware"]:
            if incident_type in df["incident_type"].values:
                subset = df[df["incident_type"] == incident_type]
                synthetic = subset[
                    subset["source"].str.contains(
                        "Synthetic|LLM|Generated", case=False, na=False
                    )
                ]
                pct = len(synthetic) / len(subset) * 100

                print(f"{incident_type}: {pct:.1f}% synthetic")

                if pct > 90:
                    print(f"  ❌ PROBLEM: Almost all synthetic!")
                    print(f"     This causes perfect accuracy (overfitting)")

    return df


def fix_dataset():
    """
    Main fix: Ensure we use VERIS data for Malware, Ransomware, DoS
    """
    print("\n" + "=" * 70)
    print("FIX STRATEGY")
    print("=" * 70)
    print()

    print("The perfect 1.00 accuracy is caused by:")
    print("  1. Too much LLM-generated synthetic data")
    print("  2. Model memorizing templates instead of learning patterns")
    print("  3. Lack of real-world variation")
    print()

    print("Solutions (in order of effectiveness):")
    print()

    print("SOLUTION 1: Use VERIS real data ⭐ RECOMMENDED")
    print("-" * 70)
    print("VERIS has ~2000 Malware, ~400 Ransomware, ~200 DoS incidents")
    print()
    print("Steps:")
    print("  1. cd Phases/Phase6")
    print("  2. python download_veris_incidents.py")
    print("     → This extracts REAL incidents from VCDB")
    print("  3. python merge_all_datasets.py")
    print("     → Merges with CERT/Enron data")
    print("  4. python ../Phase5/train_and_evaluate_pipeline.py")
    print("     → Retrain with real data")
    print()
    print("Expected result: 90-93% accuracy (realistic)")
    print()

    print("SOLUTION 2: Add noise to synthetic data")
    print("-" * 70)
    print("If you must use synthetic, add variation:")
    print("  - Use different LLM temperatures (0.7-1.2)")
    print("  - Mix multiple generation approaches")
    print("  - Add real incidents as seeds")
    print()
    print("This is NOT recommended - still causes overfitting")
    print()

    print("SOLUTION 3: Reduce dataset size")
    print("-" * 70)
    print("Smaller datasets with diverse real data > Large synthetic datasets")
    print("  - 200 real incidents per class > 600 synthetic")
    print()

    print("=" * 70)
    print("IMMEDIATE ACTION")
    print("=" * 70)
    print()
    print("Run these commands now:")
    print()
    print("  cd Phases/Phase6")
    print("  python download_veris_incidents.py")
    print("  python merge_all_datasets.py")
    print("  cd ../Phase5")
    print("  python train_and_evaluate_pipeline.py")
    print()
    print("This will:")
    print("  ✓ Replace synthetic Malware/Ransomware with VERIS real data")
    print("  ✓ Get real DoS incidents from VERIS")
    print("  ✓ Balance with existing CERT/Enron data")
    print("  ✓ Achieve realistic 90-93% accuracy")


if __name__ == "__main__":
    df = check_current_dataset()

    if df is not None:
        fix_dataset()
