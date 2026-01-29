"""
Complete Training and Evaluation Pipeline
Trains all enhanced models and generates comprehensive evaluation reports
FIXED: Resolves import issues and supports all 6 incident types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import sys
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# ================================================================
# FIX IMPORT PATHS
# ================================================================

# Get project root directory
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Try multiple import strategies
ENHANCED_AVAILABLE = False
try:
    # Strategy 1: Import from Phases.Phase5
    from Phases.Phase5.enhanced_classifier import EnhancedClassifier
    from Phases.Phase5.enhanced_similarity import EnhancedSimilarityRecommender

    ENHANCED_AVAILABLE = True
    print("✓ Imported enhanced modules from Phases.Phase5")
except ImportError:
    try:
        # Strategy 2: Import from current directory
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        from enhanced_classifier import EnhancedClassifier
        from enhanced_similarity import EnhancedSimilarityRecommender

        ENHANCED_AVAILABLE = True
        print("✓ Imported enhanced modules from current directory")
    except ImportError:
        print("=" * 70)
        print("⚠️  ENHANCED MODULES NOT FOUND")
        print("=" * 70)
        print("\nCannot import enhanced_classifier.py or enhanced_similarity.py")
        print("\nPlease ensure:")
        print("  1. Files exist in Phases/Phase5/:")
        print("     - enhanced_classifier.py")
        print("     - enhanced_similarity.py")
        print("  2. You're running from project root:")
        print(f"     Current: {Path.cwd()}")
        print(f"     Expected: {project_root}")
        print("\nAlternatively, use the simple training script:")
        print("  python train_6_class_model.py")
        print()


class ComprehensiveEvaluator:
    """
    Comprehensive model evaluation and comparison
    """

    def __init__(self, output_dir="./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def load_data(self, data_path):
        """Load and split data"""
        print("Loading data...")

        # Try multiple possible paths
        possible_paths = [
            data_path,
            f"./data/{Path(data_path).name}",
            f"../../data/{Path(data_path).name}",
            str(project_root / "data" / Path(data_path).name),
        ]

        df = None
        actual_path = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                actual_path = path
                break

        if df is None:
            raise FileNotFoundError(
                f"Could not find data file.\n"
                f"Tried:\n  " + "\n  ".join(possible_paths) + "\n\n"
                f"Please run: python expand_dataset.py"
            )

        print(f"  ✓ Loaded from: {actual_path}")

        # Check for all 6 incident types
        incident_types = df["incident_type"].unique()
        expected_types = [
            "Phishing",
            "Malware",
            "Ransomware",
            "Data Breach",
            "Insider Misuse",
            "Denial of Service",
        ]

        print(f"\n  Incident types in dataset: {len(incident_types)}")
        print(f"  Types: {sorted(incident_types)}")

        missing_types = set(expected_types) - set(incident_types)
        if missing_types:
            print(f"\n  ⚠️  WARNING: Missing incident types: {missing_types}")
            print(f"  Run expand_dataset.py to add: Malware, Ransomware, DoS")

        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df["incident_type"],
            test_size=0.2,
            random_state=42,
            stratify=df["incident_type"],
        )

        print(f"\n  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print()

        return X_train, X_test, y_train, y_test, df

    def train_models(self, X_train, y_train):
        """Train multiple model variants"""
        models = {}

        print("=" * 60)
        print("TRAINING MODELS")
        print("=" * 60 + "\n")

        # TF-IDF Ensemble (always train this)
        print("1. Training TF-IDF Ensemble Model...")
        print("   This is a fast, high-performance ensemble of:")
        print("   - Logistic Regression")
        print("   - Linear SVM")
        print("   - Multinomial Naive Bayes")
        print()

        tfidf_model = EnhancedClassifier(strategy="ensemble")
        tfidf_model.fit(X_train, y_train)
        models["tfidf_ensemble"] = tfidf_model
        print("✓ TF-IDF Ensemble Complete\n")

        # BERT (optional - commented out by default as it's slow)
        # Uncomment below to also train BERT model
        """
        print("2. Training BERT Model...")
        print("   This will take 15-30 minutes...")
        print("   Using: distilbert-base-uncased")
        print()
        
        bert_model = EnhancedClassifier(strategy='bert')
        bert_model.fit(X_train, y_train)
        models['bert'] = bert_model
        print("✓ BERT Complete\n")
        """

        return models

    def evaluate_classification(self, models, X_test, y_test):
        """Comprehensive classification evaluation"""
        print("=" * 60)
        print("CLASSIFICATION EVALUATION")
        print("=" * 60 + "\n")

        for name, model in models.items():
            print(f"Evaluating {name}...")

            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted"
            )

            # Store results
            self.results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predictions": y_pred,
                "probabilities": y_proba,
                "true_labels": y_test,
            }

            # Print summary
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}\n")

            # Detailed report
            print("Classification Report:")
            print(classification_report(y_test, y_pred, digits=3))
            print("\n" + "=" * 60 + "\n")

    def plot_confusion_matrices(self, models, X_test, y_test):
        """Generate confusion matrices for all models"""
        print("Generating confusion matrices...")

        labels = sorted(y_test.unique())
        n_models = len(models)

        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))
        if n_models == 1:
            axes = [axes]

        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            # Normalize
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            # Plot
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[idx],
                cbar_kws={"label": "Proportion"},
            )

            axes[idx].set_title(f"{name}\nNormalized Confusion Matrix", fontsize=14)
            axes[idx].set_ylabel("True Label", fontsize=12)
            axes[idx].set_xlabel("Predicted Label", fontsize=12)
            plt.setp(axes[idx].get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"  ✓ Saved to {self.output_dir / 'confusion_matrices.png'}\n")

    def plot_confidence_distribution(self, models, X_test):
        """Plot confidence score distributions"""
        print("Generating confidence distributions...")

        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for idx, (name, model) in enumerate(models.items()):
            y_proba = model.predict_proba(X_test)
            max_probs = np.max(y_proba, axis=1)

            axes[idx].hist(max_probs, bins=30, edgecolor="black", alpha=0.7)
            axes[idx].axvline(
                np.mean(max_probs),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(max_probs):.3f}",
            )
            axes[idx].axvline(
                np.median(max_probs),
                color="green",
                linestyle="--",
                label=f"Median: {np.median(max_probs):.3f}",
            )

            axes[idx].set_title(f"{name}\nConfidence Distribution", fontsize=14)
            axes[idx].set_xlabel("Confidence Score", fontsize=12)
            axes[idx].set_ylabel("Frequency", fontsize=12)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confidence_distributions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"  ✓ Saved to {self.output_dir / 'confidence_distributions.png'}\n")

    def compare_models(self):
        """Generate model comparison chart"""
        print("Generating model comparison...")

        metrics = ["accuracy", "precision", "recall", "f1"]
        model_names = list(self.results.keys())

        # Prepare data
        data = {
            metric: [self.results[m][metric] for m in model_names] for metric in metrics
        }

        # Plot
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)

        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, model_name in enumerate(model_names):
            values = [data[metric][idx] for metric in metrics]
            offset = width * idx - (width * len(model_names) / 2)
            ax.bar(x + offset, values, width, label=model_name, alpha=0.8)

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "model_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"  ✓ Saved to {self.output_dir / 'model_comparison.png'}\n")

    def generate_report(self):
        """Generate comprehensive text report"""
        print("Generating evaluation report...")

        report_path = (
            self.output_dir
            / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("INCIDENT RESPONSE MODEL EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for model_name, results in self.results.items():
                f.write("-" * 70 + "\n")
                f.write(f"MODEL: {model_name.upper()}\n")
                f.write("-" * 70 + "\n\n")

                f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall:    {results['recall']:.4f}\n")
                f.write(f"F1 Score:  {results['f1']:.4f}\n\n")

                # Confidence statistics
                max_probs = np.max(results["probabilities"], axis=1)
                f.write("Confidence Statistics:\n")
                f.write(f"  Mean:   {np.mean(max_probs):.4f}\n")
                f.write(f"  Median: {np.median(max_probs):.4f}\n")
                f.write(f"  Std:    {np.std(max_probs):.4f}\n")
                f.write(f"  Min:    {np.min(max_probs):.4f}\n")
                f.write(f"  Max:    {np.max(max_probs):.4f}\n\n")

        print(f"  ✓ Saved to {report_path}\n")

    def save_predictions(self, X_test):
        """Save predictions for error analysis"""
        print("Saving predictions for error analysis...")

        for model_name, results in self.results.items():
            df_pred = pd.DataFrame(
                {
                    "text": X_test.values,
                    "true_label": results["true_labels"].values,
                    "predicted_label": results["predictions"],
                    "confidence": np.max(results["probabilities"], axis=1),
                    "correct": results["true_labels"].values == results["predictions"],
                }
            )

            # Save all predictions
            pred_path = self.output_dir / f"{model_name}_predictions.csv"
            df_pred.to_csv(pred_path, index=False)

            # Save only errors
            errors_path = self.output_dir / f"{model_name}_errors.csv"
            df_pred[~df_pred["correct"]].to_csv(errors_path, index=False)

            print(
                f"  ✓ {model_name}: {len(df_pred)} predictions, {(~df_pred['correct']).sum()} errors"
            )

        print()


def full_training_pipeline(
    data_path="./data/real_incidents_expanded.csv", output_dir="./evaluation_results"
):
    """
    Complete training and evaluation pipeline
    """
    if not ENHANCED_AVAILABLE:
        print("\n❌ Cannot proceed - enhanced modules not available.")
        print("\nPlease use the simple training script instead:")
        print("  python train_6_class_model.py")
        return

    print("\n" + "=" * 70)
    print("INCIDENT RESPONSE MODEL - COMPLETE TRAINING PIPELINE")
    print("=" * 70 + "\n")

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(output_dir)

    # Load data
    try:
        X_train, X_test, y_train, y_test, df = evaluator.load_data(data_path)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return

    # Train models
    models = evaluator.train_models(X_train, y_train)

    # Evaluate
    evaluator.evaluate_classification(models, X_test, y_test)

    # Visualizations
    evaluator.plot_confusion_matrices(models, X_test, y_test)
    evaluator.plot_confidence_distribution(models, X_test)
    evaluator.compare_models()

    # Reports
    evaluator.generate_report()
    evaluator.save_predictions(X_test)

    # Save models
    print("Saving models...")
    for name, model in models.items():
        model_path = Path(f"./models/enhanced_{name}")
        model.save(model_path)
        print(f"  ✓ Saved {name} to {model_path}")

    # Build similarity recommender
    print("\nBuilding similarity recommender...")
    recommender = EnhancedSimilarityRecommender(
        use_hybrid=True, time_decay_enabled=True
    )
    recommender.fit(df)
    recommender.save("./models/enhanced_similarity")
    print("  ✓ Similarity recommender saved\n")

    print("=" * 70)
    print("✓ TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review evaluation results in ./evaluation_results/")
    print("  2. Check error analysis in *_errors.csv files")
    print("  3. Restart backend: python -m uvicorn api.server:app --reload")
    print("  4. Test with frontend: cd frontend && npm start")
    print()


if __name__ == "__main__":
    # Determine which data file to use
    data_paths = [
        "./data/real_incidents_expanded.csv",  # Preferred (6 types)
        "../../data/real_incidents_expanded.csv",
        "./data/real_incidents_balanced.csv",  # Fallback (3 types)
        "../../data/real_incidents_balanced.csv",
    ]

    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break

    if data_path is None:
        print("=" * 70)
        print("❌ NO TRAINING DATA FOUND")
        print("=" * 70)
        print("\nCannot find training data file.")
        print("\nPlease run: python expand_dataset.py")
        print("\nThis will create: ./data/real_incidents_expanded.csv")
        print("with all 6 incident types (Phishing, Malware, Ransomware,")
        print("Data Breach, Insider Misuse, Denial of Service)")
        sys.exit(1)

    # Run pipeline
    full_training_pipeline(data_path=data_path)

    # Test the trained model
    if ENHANCED_AVAILABLE:
        print("\n" + "=" * 70)
        print("TESTING TRAINED MODEL")
        print("=" * 70 + "\n")

        try:
            model = EnhancedClassifier.load("./models/enhanced_tfidf_ensemble")

            test_cases = {
                "Phishing": "Employee clicked suspicious email link and entered credentials",
                "Malware": "Endpoint protection detected malicious executable with elevated privileges",
                "Ransomware": "Critical alert: Files encrypted on server. Ransom note demanding Bitcoin payment",
                "Data Breach": "Unusual database access from admin account. Large data transfer to external IP",
                "Insider Misuse": "Employee accessing files outside job scope during after-hours",
                "Denial of Service": "Network reports severe bandwidth saturation. DDoS attack from 10,000 IPs",
            }

            print("Test Predictions:\n")
            all_correct = True
            for expected, incident in test_cases.items():
                pred = model.predict([incident])[0]
                proba = model.predict_proba([incident])[0]
                conf = max(proba)

                status = "✓" if pred == expected else "❌"
                print(f"{status} {expected:20s} → {pred:20s} ({conf:.1%})")

                if pred != expected:
                    all_correct = False

            print()
            if all_correct:
                print("✅ All test predictions correct!")
            else:
                print(
                    "⚠️  Some predictions incorrect - model may need more training data"
                )
        except Exception as e:
            print(f"⚠️  Could not load model for testing: {e}")
