"""
Diagnose and fix data leakage issues in incident dataset

This script:
1. Detects if data is too uniform (overfitting indicator)
2. Checks for synthetic patterns
3. Suggests fixes
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_dataset_quality(csv_path):
    """
    Analyze dataset for signs of data leakage or overfitting
    """
    print("=" * 70)
    print("DATASET QUALITY ANALYSIS")
    print("=" * 70)
    print()

    df = pd.read_csv(csv_path)

    # 1. Check source distribution
    print("1. SOURCE DISTRIBUTION")
    print("-" * 70)
    if "source" in df.columns:
        print(df.groupby(["incident_type", "source"]).size().unstack(fill_value=0))
        print()

        # Flag if too much synthetic
        for incident_type in df["incident_type"].unique():
            subset = df[df["incident_type"] == incident_type]
            synthetic_count = subset[
                subset["source"].str.contains(
                    "Synthetic|LLM|Generated", case=False, na=False
                )
            ].shape[0]
            total_count = len(subset)
            synthetic_pct = (synthetic_count / total_count) * 100

            print(f"{incident_type}: {synthetic_pct:.1f}% synthetic")
            if synthetic_pct > 80:
                print(f"  ⚠️  WARNING: Too much synthetic data!")

    print()

    # 2. Text diversity analysis
    print("2. TEXT DIVERSITY ANALYSIS")
    print("-" * 70)

    for incident_type in df["incident_type"].unique():
        subset = df[df["incident_type"] == incident_type]["text"]

        if len(subset) < 2:
            continue

        # Calculate average similarity within class
        vectorizer = TfidfVectorizer(max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(
                subset.head(100)
            )  # Sample for speed
            similarities = cosine_similarity(tfidf_matrix)

            # Get upper triangle (exclude diagonal)
            upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
            avg_similarity = upper_tri.mean()

            print(f"{incident_type}:")
            print(f"  Avg intra-class similarity: {avg_similarity:.3f}")

            if avg_similarity > 0.5:
                print(
                    f"  ⚠️  WARNING: Very high similarity (likely synthetic/template-based)"
                )
            elif avg_similarity > 0.3:
                print(f"  ⚠️  CAUTION: Moderate similarity (check for templates)")
            else:
                print(f"  ✓ Good diversity")
        except:
            print(f"{incident_type}: Unable to analyze")

    print()

    # 3. Vocabulary overlap
    print("3. VOCABULARY OVERLAP")
    print("-" * 70)

    class_vocab = {}
    for incident_type in df["incident_type"].unique():
        subset = df[df["incident_type"] == incident_type]["text"]

        # Get top words
        vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
        try:
            vectorizer.fit(subset)
            class_vocab[incident_type] = set(vectorizer.get_feature_names_out())
        except:
            class_vocab[incident_type] = set()

    # Check overlap between classes
    incident_types = list(class_vocab.keys())
    for i, type1 in enumerate(incident_types):
        for type2 in incident_types[i + 1 :]:
            overlap = class_vocab[type1] & class_vocab[type2]
            overlap_pct = (
                len(overlap)
                / min(len(class_vocab[type1]), len(class_vocab[type2]))
                * 100
            )

            if overlap_pct < 30:
                print(
                    f"{type1} vs {type2}: {overlap_pct:.1f}% overlap ⚠️  (Too distinct - possible leakage)"
                )
            elif overlap_pct > 70:
                print(
                    f"{type1} vs {type2}: {overlap_pct:.1f}% overlap (Might be hard to distinguish)"
                )
            else:
                print(f"{type1} vs {type2}: {overlap_pct:.1f}% overlap ✓")

    print()

    # 4. Check for obvious markers
    print("4. CHECKING FOR SYNTHETIC MARKERS")
    print("-" * 70)

    markers = {
        "Template indicators": [
            "botnet comprising",
            "network operations center detected",
            "attack targeted",
        ],
        "LLM artifacts": ["approximately", "sustained for", "across X countries"],
        "Unnatural patterns": ["exactly", "precisely", "specifically"],
    }

    for marker_type, patterns in markers.items():
        print(f"\n{marker_type}:")
        for pattern in patterns:
            count = df["text"].str.contains(pattern, case=False, na=False).sum()
            pct = (count / len(df)) * 100
            if pct > 20:
                print(f"  '{pattern}': {pct:.1f}% of incidents ⚠️")

    print()

    # 5. Recommendations
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()

    # Calculate overall synthetic percentage
    if "source" in df.columns:
        synthetic_total = df[
            df["source"].str.contains("Synthetic|LLM|Generated", case=False, na=False)
        ].shape[0]
        synthetic_pct_total = (synthetic_total / len(df)) * 100

        if synthetic_pct_total > 60:
            print("❌ CRITICAL: Over 60% synthetic data")
            print("   → Replace with VERIS real data")
            print("   → Run: python download_veris_incidents.py")
            print()
        elif synthetic_pct_total > 40:
            print("⚠️  WARNING: Over 40% synthetic data")
            print("   → Add more real incidents from VERIS")
            print()
        else:
            print("✓ Acceptable synthetic ratio")
            print()

    # Check perfect accuracy classes
    print("For classes showing 1.00 accuracy:")
    print("  1. Verify they have real data (not 100% synthetic)")
    print("  2. Check text diversity (should be < 0.4 avg similarity)")
    print("  3. Ensure vocabulary overlap with other classes (30-70%)")
    print()
    print("Quick fixes:")
    print("  → Run collect_ddos_veris_llm.py (gets ~200 real DoS from VERIS)")
    print("  → Ensure VERIS has Malware and Ransomware data")
    print("  → Downsample if one source dominates")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "data/real_incidents_balanced.csv"

    analyze_dataset_quality(csv_path)
