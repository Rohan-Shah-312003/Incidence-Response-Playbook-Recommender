# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report
# import numpy as np
# import os

# # --- Configuration ---
# # Path to your predictions file
# DATA_PATH = "evaluation_results/tfidf_ensemble_predictions.csv"
# OUTPUT_DIR = "evaluation_results"

# # Ensure output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Set publication-quality plot style
# sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["savefig.dpi"] = 300


# def load_data(path):
#     """Loads the prediction CSV and ensures correct data types."""
#     if not os.path.exists(path):
#         print(f"Error: File not found at {path}")
#         return None

#     df = pd.read_csv(path)
#     # Ensure correct boolean type
#     df["correct"] = df["true_label"] == df["predicted_label"]
#     return df


# def plot_confusion_matrix(df):
#     """Generates a normalized confusion matrix heatmap."""
#     labels = sorted(df["true_label"].unique())
#     cm = confusion_matrix(df["true_label"], df["predicted_label"], labels=labels)

#     # Normalize by row (true class)
#     cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

#     plt.figure(figsize=(8, 6))
#     heatmap = sns.heatmap(
#         cm_norm,
#         annot=True,
#         fmt=".2f",
#         cmap="Blues",
#         xticklabels=labels,
#         yticklabels=labels,
#         cbar_kws={"label": "Proportion"},
#     )

#     plt.title("Confusion Matrix (Normalized)", fontsize=14, pad=20)
#     plt.ylabel("True Label", fontweight="bold")
#     plt.xlabel("Predicted Label", fontweight="bold")
#     plt.tight_layout()
#     plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
#     plt.close()
#     print("Generated Confusion Matrix.")


# def plot_confidence_distribution(df):
#     """
#     Plots the distribution of confidence scores for Correct vs Incorrect predictions.
#     Critical for analyzing model calibration.
#     """
#     plt.figure(figsize=(10, 6))

#     sns.kdeplot(
#         data=df[df["correct"] == True],
#         x="confidence",
#         fill=True,
#         color="green",
#         label="Correct Predictions",
#         alpha=0.3,
#     )
#     sns.kdeplot(
#         data=df[df["correct"] == False],
#         x="confidence",
#         fill=True,
#         color="red",
#         label="Incorrect Predictions",
#         alpha=0.3,
#     )

#     plt.title(
#         "Confidence Score Distribution: Correct vs. Incorrect", fontsize=14, pad=20
#     )
#     plt.xlabel("Model Confidence Score")
#     plt.ylabel("Density")
#     plt.legend(loc="upper left")
#     plt.xlim(0, 1.0)

#     plt.tight_layout()
#     plt.savefig(f"{OUTPUT_DIR}/confidence_distribution.png")
#     plt.close()
#     print("Generated Confidence Distribution Plot.")


# def plot_class_metrics(df):
#     """Generates a grouped bar chart for Precision, Recall, and F1-Score."""
#     report = classification_report(
#         df["true_label"], df["predicted_label"], output_dict=True
#     )

#     # Convert to DataFrame and drop 'accuracy', 'macro avg', etc.
#     metrics_df = pd.DataFrame(report).transpose()
#     classes = [
#         idx
#         for idx in metrics_df.index
#         if idx not in ["accuracy", "macro avg", "weighted avg"]
#     ]
#     metrics_df = metrics_df.loc[classes, ["precision", "recall", "f1-score"]]

#     # Melt for seaborn plotting
#     metrics_melted = metrics_df.reset_index().melt(
#         id_vars="index", var_name="Metric", value_name="Score"
#     )

#     plt.figure(figsize=(10, 6))
#     bar_plot = sns.barplot(
#         data=metrics_melted, x="index", y="Score", hue="Metric", palette="viridis"
#     )

#     plt.title("Performance Metrics by Class", fontsize=14, pad=20)
#     plt.xlabel("Class Label", fontweight="bold")
#     plt.ylabel("Score")
#     plt.ylim(0, 1.05)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

#     # Add values on top of bars
#     for container in bar_plot.containers:
#         bar_plot.bar_label(container, fmt="%.2f", padding=3, fontsize=10)

#     plt.tight_layout()
#     plt.savefig(f"{OUTPUT_DIR}/class_performance_metrics.png")
#     plt.close()
#     print("Generated Class Metrics Plot.")


# def plot_confidence_boxplot(df):
#     """Boxplot showing confidence ranges for each class, split by correctness."""
#     plt.figure(figsize=(12, 6))

#     sns.boxplot(
#         data=df,
#         x="true_label",
#         y="confidence",
#         hue="correct",
#         palette={True: "mediumseagreen", False: "indianred"},
#         showfliers=False,
#     )

#     plt.title("Confidence Levels by Class and Correctness", fontsize=14, pad=20)
#     plt.xlabel("Class Label", fontweight="bold")
#     plt.ylabel("Confidence Score")
#     plt.legend(title="Prediction Status")

#     plt.tight_layout()
#     plt.savefig(f"{OUTPUT_DIR}/confidence_boxplot.png")
#     plt.close()
#     print("Generated Confidence Boxplot.")


# def main():
#     print("Loading data...")
#     df = load_data(DATA_PATH)

#     if df is not None:
#         plot_confusion_matrix(df)
#         plot_confidence_distribution(df)
#         plot_class_metrics(df)
#         plot_confidence_boxplot(df)
#         print(f"\nAll plots saved to directory: {OUTPUT_DIR}")


# if __name__ == "__main__":
#     main()


"""
Standalone Plotting Script for Phase5
Run this directly after training to generate all plots

Usage:
    cd Phases/Phase5
    python generate_plots_standalone.py
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

# Add parent directory to path to import train_models
sys.path.append(str(Path(__file__).parent))

from train_models import train
from preprocess import load_data, split_data

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = "./Phase5/results"


def ensure_output_dir():
    """Create output directory"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Proportion'},
        square=True,
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title(f'{model_name}\nNormalized Confusion Matrix',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = f"{OUTPUT_DIR}/{model_name}_confusion_matrix.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {save_path}")


def plot_per_class_metrics(y_true, y_pred, labels, model_name):
    """Bar chart of metrics"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None
    )
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax.bar(x, recall, width, label='Recall', color='lightgreen')
    ax.bar(x + width, f1, width, label='F1-Score', color='salmon')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'{model_name}\nPer-Class Performance',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f"{OUTPUT_DIR}/{model_name}_metrics.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {save_path}")


def plot_confidence_distribution(y_proba, y_true, y_pred, model_name):
    """KDE of confidence scores"""
    confidences = np.max(y_proba, axis=1)
    correct = y_true == y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Correct
    sns.kdeplot(
        confidences[correct],
        fill=True,
        color='green',
        label='Correct',
        alpha=0.4,
        linewidth=2,
        ax=ax
    )
    
    # Incorrect
    if (~correct).sum() > 0:
        sns.kdeplot(
            confidences[~correct],
            fill=True,
            color='red',
            label='Incorrect',
            alpha=0.4,
            linewidth=2,
            ax=ax
        )
    
    ax.set_title(f'{model_name}\nConfidence Distribution',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Confidence', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f"{OUTPUT_DIR}/{model_name}_confidence.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {save_path}")


def generate_text_report(models, X_test, y_test):
    """Text summary"""
    report_path = f"{OUTPUT_DIR}/evaluation_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("IRPR EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        for name, pipeline in models.items():
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)
            
            f.write("=" * 80 + "\n")
            f.write(f"{name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(classification_report(y_test, y_pred))
            f.write("\n")
            
            confidences = np.max(y_proba, axis=1)
            f.write(f"Confidence: mean={confidences.mean():.3f}, "
                   f"median={confidences.median():.3f}\n\n")
    
    print(f"\n✓ {report_path}")


def main():
    """Main execution"""
    print("=" * 80)
    print("IRPR - GENERATE EVALUATION PLOTS")
    print("=" * 80)
    print()
    
    ensure_output_dir()
    
    # Train or load models
    print("Training models...")
    models, X_test, y_test = train()
    
    print("\nGenerating plots...")
    print("-" * 80)
    
    for name, pipeline in models.items():
        print(f"\n{name}:")
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        labels = pipeline.classes_
        
        # Generate plots
        plot_confusion_matrix(y_test, y_pred, labels, name)
        plot_per_class_metrics(y_test, y_pred, labels, name)
        plot_confidence_distribution(y_proba, y_test, y_pred, name)
        
        # Save predictions
        results_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'confidence': np.max(y_proba, axis=1),
            'correct': y_test == y_pred
        })
        
        pred_path = f"{OUTPUT_DIR}/{name}_predictions.csv"
        results_df.to_csv(pred_path, index=False)
        print(f"  ✓ {pred_path}")
    
    # Generate report
    generate_text_report(models, X_test, y_test)
    
    print("\n" + "=" * 80)
    print("✓ ALL PLOTS GENERATED")
    print("=" * 80)
    print(f"\nOutput: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()