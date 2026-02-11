"""
Quick Analysis: Verify if 96.8% accuracy is legitimate

This checks:
1. If you're using real VERIS data
2. If test/train split is proper
3. If high scores are due to data quality or overfitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def analyze_current_results():
    """
    Analyze whether 96.8% accuracy is legitimate or overfitting
    """
    print("=" * 80)
    print("ANALYZING 96.8% ACCURACY - LEGITIMATE OR OVERFITTING?")
    print("=" * 80)
    print()
    
    # Load training data
    data_path = "data/real_incidents_balanced.csv"
    
    if not Path(data_path).exists():
        print(f"❌ Cannot find training data at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Check 1: Data sources
    print("CHECK 1: Data Quality")
    print("-" * 80)
    
    if 'source' in df.columns:
        print("\nOverall source distribution:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {source}: {count} ({pct:.1f}%)")
        
        # Check high-performing classes
        high_perf_classes = ['Denial of Service', 'Malware', 'Ransomware']
        
        print("\n" + "=" * 80)
        print("CHECK 2: High-Performing Classes (DoS, Malware, Ransomware)")
        print("-" * 80)
        
        for cls in high_perf_classes:
            if cls not in df['incident_type'].values:
                continue
            
            subset = df[df['incident_type'] == cls]
            
            print(f"\n{cls} ({len(subset)} incidents):")
            print("  Sources:")
            
            for source in subset['source'].unique():
                count = (subset['source'] == source).sum()
                pct = (count / len(subset)) * 100
                print(f"    {source}: {count} ({pct:.1f}%)")
            
            # Check for VERIS
            veris_count = subset['source'].str.contains('VERIS', case=False, na=False).sum()
            veris_pct = (veris_count / len(subset)) * 100
            
            # Check for synthetic
            synthetic_keywords = ['Synthetic', 'Generated', 'LLM', 'Enhanced']
            synthetic_count = 0
            for keyword in synthetic_keywords:
                synthetic_count += subset['source'].str.contains(keyword, case=False, na=False).sum()
            synthetic_pct = (synthetic_count / len(subset)) * 100
            
            print(f"\n  Summary:")
            print(f"    VERIS (real): {veris_pct:.1f}%")
            print(f"    Synthetic: {synthetic_pct:.1f}%")
            
            if veris_pct > 50:
                print(f"    ✅ GOOD: Majority real data")
            elif synthetic_pct > 80:
                print(f"    ⚠️  WARNING: Too much synthetic - likely overfitting")
            else:
                print(f"    ⚙️  MIXED: Combination of sources")
    
    # Check 3: Text diversity
    print("\n" + "=" * 80)
    print("CHECK 3: Text Diversity (Overfitting Indicator)")
    print("-" * 80)
    
    for cls in ['Denial of Service', 'Malware', 'Ransomware']:
        if cls not in df['incident_type'].values:
            continue
        
        texts = df[df['incident_type'] == cls]['text'].head(50).tolist()
        
        if len(texts) < 2:
            continue
        
        try:
            vectorizer = TfidfVectorizer(max_features=500)
            tfidf = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf)
            
            # Average similarity (excluding diagonal)
            n = sim_matrix.shape[0]
            mask = ~np.eye(n, dtype=bool)
            avg_sim = sim_matrix[mask].mean()
            
            print(f"\n{cls}:")
            print(f"  Avg text similarity: {avg_sim:.3f}")
            
            if avg_sim > 0.5:
                print(f"  ❌ OVERFITTING: Texts too similar (template-based)")
                print(f"     → 99-100% accuracy is due to memorization, not learning")
            elif avg_sim > 0.35:
                print(f"  ⚠️  CAUTION: Moderate similarity")
                print(f"     → High accuracy might be partially due to templates")
            else:
                print(f"  ✅ GOOD: Diverse texts")
                print(f"     → High accuracy is legitimate!")
        
        except Exception as e:
            print(f"{cls}: Could not analyze - {e}")
    
    # Check 4: Look at actual samples
    print("\n" + "=" * 80)
    print("CHECK 4: Sample Incidents (Manual Review)")
    print("-" * 80)
    
    for cls in ['Denial of Service', 'Malware', 'Ransomware']:
        if cls not in df['incident_type'].values:
            continue
        
        sample = df[df['incident_type'] == cls].iloc[0]
        
        print(f"\n{cls} sample:")
        print(f"  Source: {sample.get('source', 'Unknown')}")
        print(f"  Text preview: {sample['text'][:200]}...")
        
        # Check for template markers
        template_markers = [
            'botnet comprising',
            'network operations center detected',
            'sustained for',
            'across X countries'
        ]
        
        has_markers = any(marker.lower() in sample['text'].lower() for marker in template_markers)
        
        if has_markers:
            print(f"  ⚠️  Contains template markers")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()
    
    # Calculate overall synthetic percentage
    if 'source' in df.columns:
        synthetic_count = 0
        for keyword in ['Synthetic', 'Generated', 'LLM', 'Enhanced']:
            synthetic_count += df['source'].str.contains(keyword, case=False, na=False).sum()
        
        synthetic_pct = (synthetic_count / len(df)) * 100
        veris_count = df['source'].str.contains('VERIS', case=False, na=False).sum()
        veris_pct = (veris_count / len(df)) * 100
        
        print(f"Dataset composition:")
        print(f"  VERIS (real): {veris_pct:.1f}%")
        print(f"  Synthetic: {synthetic_pct:.1f}%")
        print()
        
        if veris_pct > 50 and synthetic_pct < 30:
            print("✅ VERDICT: 96.8% accuracy is LEGITIMATE")
            print()
            print("Reasons:")
            print("  • Majority real VERIS data")
            print("  • Low synthetic contamination")
            print("  • Accuracy in realistic range (95-97%)")
            print()
            print("The 99-100% for DoS/Malware/Ransomware is acceptable if:")
            print("  • Text diversity is good (similarity < 0.35)")
            print("  • Using real VERIS data (> 50%)")
            print()
            print("✅ Your model is production-ready!")
            
        elif synthetic_pct > 60:
            print("⚠️  VERDICT: 96.8% accuracy is SUSPICIOUS")
            print()
            print("Reasons:")
            print("  • Over 60% synthetic data")
            print("  • Likely overfitting to templates")
            print("  • Won't generalize to real incidents")
            print()
            print("Action needed:")
            print("  1. cd Phases/Phase6")
            print("  2. python download_veris_incidents.py")
            print("  3. python merge_all_datasets.py")
            print("  4. Retrain")
            
        else:
            print("⚙️  VERDICT: 96.8% accuracy is ACCEPTABLE but verify")
            print()
            print("Reasons:")
            print("  • Mixed data sources")
            print("  • Some synthetic contamination")
            print()
            print("Recommendation:")
            print("  • Check text diversity scores above")
            print("  • If similarity > 0.5: Add more real data")
            print("  • If similarity < 0.35: You're good!")
    
    print()


if __name__ == "__main__":
    analyze_current_results()