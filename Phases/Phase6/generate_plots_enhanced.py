"""
Enhanced Plotting Script for IRPR Model Evaluation

Generates comprehensive visualizations:
1. Confusion Matrix (Normalized)
2. Per-Class Performance (Precision, Recall, F1)
3. Confidence Score Distribution
4. Confidence vs Correctness
5. Error Analysis
6. Learning Curves (if available)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Configuration
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

# Paths
PREDICTIONS_PATH = "evaluation_results/tfidf_ensemble_predictions.csv"
OUTPUT_DIR = "evaluation_results/plots"


def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_predictions():
    """Load prediction results"""
    if not Path(PREDICTIONS_PATH).exists():
        print(f"❌ Predictions file not found: {PREDICTIONS_PATH}")
        print("\nMake sure you've run train_and_evaluate_pipeline.py first")
        return None

    df = pd.read_csv(PREDICTIONS_PATH)

    # Verify required columns
    required_cols = ["true_label", "predicted_label", "confidence"]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Missing required columns. Found: {df.columns.tolist()}")
        return None

    # Add correctness column if not present
    if "correct" not in df.columns:
        df["correct"] = df["true_label"] == df["predicted_label"]

    print(f"✓ Loaded {len(df)} predictions")
    return df


def plot_confusion_matrix(df, save_path):
    """
    Generate normalized confusion matrix heatmap
    """
    print("Generating confusion matrix...")

    labels = sorted(df["true_label"].unique())
    cm = confusion_matrix(df["true_label"], df["predicted_label"], labels=labels)

    # Normalize by row (true class)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Proportion"},
        square=True,
        linewidths=0.5,
        ax=ax,
    )

    # Styling
    ax.set_title(
        "Normalized Confusion Matrix\n(Row-normalized)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {save_path}")


def plot_per_class_metrics(df, save_path):
    """
    Bar chart showing precision, recall, F1 for each class
    """
    print("Generating per-class metrics...")

    labels = sorted(df["true_label"].unique())

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["true_label"], df["predicted_label"], labels=labels, average=None
    )

    # Create DataFrame for plotting
    metrics_df = pd.DataFrame(
        {"Class": labels, "Precision": precision, "Recall": recall, "F1-Score": f1}
    )

    # Melt for grouped bar chart
    metrics_melted = metrics_df.melt(
        id_vars="Class", var_name="Metric", value_name="Score"
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_plot = sns.barplot(
        data=metrics_melted, x="Class", y="Score", hue="Metric", palette="Set2", ax=ax
    )

    # Add value labels on bars
    for container in bar_plot.containers:
        bar_plot.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    # Styling
    ax.set_title(
        "Per-Class Performance Metrics", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_xlabel("Incident Type", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Metric", loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {save_path}")


def plot_confidence_distribution(df, save_path):
    """
    Distribution of confidence scores for correct vs incorrect predictions
    """
    print("Generating confidence distribution...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # KDE plots
    sns.kdeplot(
        data=df[df["correct"] == True],
        x="confidence",
        fill=True,
        color="green",
        label="Correct Predictions",
        alpha=0.4,
        linewidth=2,
        ax=ax,
    )

    sns.kdeplot(
        data=df[df["correct"] == False],
        x="confidence",
        fill=True,
        color="red",
        label="Incorrect Predictions",
        alpha=0.4,
        linewidth=2,
        ax=ax,
    )

    # Add vertical lines for means
    correct_mean = df[df["correct"] == True]["confidence"].mean()
    incorrect_mean = df[df["correct"] == False]["confidence"].mean()

    ax.axvline(
        correct_mean,
        color="darkgreen",
        linestyle="--",
        label=f"Correct Mean: {correct_mean:.3f}",
    )
    ax.axvline(
        incorrect_mean,
        color="darkred",
        linestyle="--",
        label=f"Incorrect Mean: {incorrect_mean:.3f}",
    )

    # Styling
    ax.set_title(
        "Confidence Score Distribution:\nCorrect vs Incorrect Predictions",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Model Confidence Score", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="upper left", frameon=True, shadow=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {save_path}")


def plot_confidence_boxplot(df, save_path):
    """
    Boxplot of confidence scores by class and correctness
    """
    print("Generating confidence boxplot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x="true_label",
        y="confidence",
        hue="correct",
        palette={True: "lightgreen", False: "lightcoral"},
        showfliers=False,
        ax=ax,
    )

    # Styling
    ax.set_title(
        "Confidence Levels by Class and Correctness",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("True Class", fontweight="bold")
    ax.set_ylabel("Confidence Score", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Prediction", labels=["Incorrect", "Correct"])
    ax.grid(axis="y", alpha=0.3)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {save_path}")


def plot_error_analysis(df, save_path):
    """
    Heatmap showing confusion patterns (where errors occur)
    """
    print("Generating error analysis...")

    # Get only incorrect predictions
    errors = df[df["correct"] == False]

    if len(errors) == 0:
        print("  ⚠️  No errors found - perfect accuracy!")
        return

    labels = sorted(df["true_label"].unique())

    # Create error matrix
    error_matrix = pd.crosstab(
        errors["true_label"], errors["predicted_label"], normalize="index"
    ).reindex(index=labels, columns=labels, fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        error_matrix,
        annot=True,
        fmt=".1%",
        cmap="Reds",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Error Rate"},
        ax=ax,
    )

    ax.set_title(
        "Error Analysis: Misclassification Patterns\n(Percentage of errors per true class)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {save_path}")


def plot_accuracy_by_confidence(df, save_path):
    """
    Line plot showing accuracy at different confidence thresholds
    """
    print("Generating accuracy vs confidence threshold...")

    thresholds = np.arange(0.0, 1.01, 0.05)
    accuracies = []
    coverages = []

    for threshold in thresholds:
        subset = df[df["confidence"] >= threshold]

        if len(subset) > 0:
            accuracy = subset["correct"].sum() / len(subset)
            coverage = len(subset) / len(df)
        else:
            accuracy = 0
            coverage = 0

        accuracies.append(accuracy)
        coverages.append(coverage)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Accuracy line
    color = "tab:blue"
    ax1.set_xlabel("Confidence Threshold", fontweight="bold")
    ax1.set_ylabel("Accuracy", color=color, fontweight="bold")
    ax1.plot(
        thresholds,
        accuracies,
        color=color,
        linewidth=2,
        marker="o",
        markersize=4,
        label="Accuracy",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    # Coverage line (secondary axis)
    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Coverage (% of predictions)", color=color, fontweight="bold")
    ax2.plot(
        thresholds,
        coverages,
        color=color,
        linewidth=2,
        marker="s",
        markersize=4,
        label="Coverage",
        linestyle="--",
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1.05)

    # Title
    fig.suptitle(
        "Accuracy vs Confidence Threshold\n(Trade-off between accuracy and coverage)",
        fontsize=14,
        fontweight="bold",
    )

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {save_path}")


def generate_summary_report(df, save_path):
    """
    Generate text summary report
    """
    print("Generating summary report...")

    with open(save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("INCIDENT RESPONSE MODEL - EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Total predictions: {len(df)}\n\n")

        # Overall accuracy
        overall_acc = df["correct"].sum() / len(df)
        f.write(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)\n\n")

        # Per-class metrics
        f.write("=" * 80 + "\n")
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("=" * 80 + "\n\n")

        labels = sorted(df["true_label"].unique())
        precision, recall, f1, support = precision_recall_fscore_support(
            df["true_label"], df["predicted_label"], labels=labels
        )

        f.write(
            f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n"
        )
        f.write("-" * 80 + "\n")

        for i, label in enumerate(labels):
            f.write(
                f"{label:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                f"{f1[i]:<12.4f} {support[i]:<10}\n"
            )

        f.write("\n")

        # Confidence analysis
        f.write("=" * 80 + "\n")
        f.write("CONFIDENCE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Overall confidence statistics:\n")
        f.write(f"  Mean:   {df['confidence'].mean():.4f}\n")
        f.write(f"  Median: {df['confidence'].median():.4f}\n")
        f.write(f"  Std:    {df['confidence'].std():.4f}\n")
        f.write(f"  Min:    {df['confidence'].min():.4f}\n")
        f.write(f"  Max:    {df['confidence'].max():.4f}\n\n")

        correct_conf = df[df["correct"] == True]["confidence"].mean()
        incorrect_conf = df[df["correct"] == False]["confidence"].mean()

        f.write(f"Confidence by correctness:\n")
        f.write(f"  Correct predictions:   {correct_conf:.4f}\n")
        f.write(f"  Incorrect predictions: {incorrect_conf:.4f}\n")
        f.write(f"  Separation:            {correct_conf - incorrect_conf:.4f}\n\n")

        # Error analysis
        if (df["correct"] == False).sum() > 0:
            f.write("=" * 80 + "\n")
            f.write("ERROR ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            errors = df[df["correct"] == False]
            f.write(
                f"Total errors: {len(errors)} ({len(errors) / len(df) * 100:.2f}%)\n\n"
            )

            f.write("Most common misclassifications:\n")
            error_pairs = errors.groupby(["true_label", "predicted_label"]).size()
            error_pairs = error_pairs.sort_values(ascending=False).head(10)

            for (true_label, pred_label), count in error_pairs.items():
                pct = count / len(errors) * 100
                f.write(
                    f"  {true_label} → {pred_label}: {count} ({pct:.1f}% of errors)\n"
                )

    print(f"  ✓ Saved to {save_path}")


def main():
    """
    Main plotting pipeline
    """
    print("=" * 80)
    print("IRPR MODEL - COMPREHENSIVE EVALUATION PLOTS")
    print("=" * 80)
    print()

    # Setup
    ensure_output_dir()

    # Load data
    df = load_predictions()
    if df is None:
        return

    print()
    print("Generating visualizations...")
    print("-" * 80)

    # Generate plots
    plot_confusion_matrix(df, f"{OUTPUT_DIR}/01_confusion_matrix.png")
    plot_per_class_metrics(df, f"{OUTPUT_DIR}/02_per_class_metrics.png")
    plot_confidence_distribution(df, f"{OUTPUT_DIR}/03_confidence_distribution.png")
    plot_confidence_boxplot(df, f"{OUTPUT_DIR}/04_confidence_boxplot.png")
    plot_error_analysis(df, f"{OUTPUT_DIR}/05_error_analysis.png")
    plot_accuracy_by_confidence(df, f"{OUTPUT_DIR}/06_accuracy_vs_threshold.png")
    generate_summary_report(df, f"{OUTPUT_DIR}/evaluation_summary.txt")

    print()
    print("=" * 80)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print(f"Output location: {OUTPUT_DIR}/")
    print()
    print("Generated files:")
    print("  01_confusion_matrix.png          - Normalized confusion matrix")
    print("  02_per_class_metrics.png         - Precision/Recall/F1 bars")
    print("  03_confidence_distribution.png   - Confidence KDE plots")
    print("  04_confidence_boxplot.png        - Confidence by class")
    print("  05_error_analysis.png            - Error heatmap")
    print("  06_accuracy_vs_threshold.png     - Accuracy/coverage trade-off")
    print("  evaluation_summary.txt           - Text report")
    print()


if __name__ == "__main__":
    main()
