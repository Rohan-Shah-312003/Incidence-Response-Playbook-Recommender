import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# --- Configuration ---
# Path to your predictions file
DATA_PATH = "evaluation_results/tfidf_ensemble_predictions.csv"
OUTPUT_DIR = "evaluation_results"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set publication-quality plot style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def load_data(path):
    """Loads the prediction CSV and ensures correct data types."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None

    df = pd.read_csv(path)
    # Ensure correct boolean type
    df["correct"] = df["true_label"] == df["predicted_label"]
    return df


def plot_confusion_matrix(df):
    """Generates a normalized confusion matrix heatmap."""
    labels = sorted(df["true_label"].unique())
    cm = confusion_matrix(df["true_label"], df["predicted_label"], labels=labels)

    # Normalize by row (true class)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Proportion"},
    )

    plt.title("Confusion Matrix (Normalized)", fontsize=14, pad=20)
    plt.ylabel("True Label", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    plt.close()
    print("Generated Confusion Matrix.")


def plot_confidence_distribution(df):
    """
    Plots the distribution of confidence scores for Correct vs Incorrect predictions.
    Critical for analyzing model calibration.
    """
    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        data=df[df["correct"] == True],
        x="confidence",
        fill=True,
        color="green",
        label="Correct Predictions",
        alpha=0.3,
    )
    sns.kdeplot(
        data=df[df["correct"] == False],
        x="confidence",
        fill=True,
        color="red",
        label="Incorrect Predictions",
        alpha=0.3,
    )

    plt.title(
        "Confidence Score Distribution: Correct vs. Incorrect", fontsize=14, pad=20
    )
    plt.xlabel("Model Confidence Score")
    plt.ylabel("Density")
    plt.legend(loc="upper left")
    plt.xlim(0, 1.0)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confidence_distribution.png")
    plt.close()
    print("Generated Confidence Distribution Plot.")


def plot_class_metrics(df):
    """Generates a grouped bar chart for Precision, Recall, and F1-Score."""
    report = classification_report(
        df["true_label"], df["predicted_label"], output_dict=True
    )

    # Convert to DataFrame and drop 'accuracy', 'macro avg', etc.
    metrics_df = pd.DataFrame(report).transpose()
    classes = [
        idx
        for idx in metrics_df.index
        if idx not in ["accuracy", "macro avg", "weighted avg"]
    ]
    metrics_df = metrics_df.loc[classes, ["precision", "recall", "f1-score"]]

    # Melt for seaborn plotting
    metrics_melted = metrics_df.reset_index().melt(
        id_vars="index", var_name="Metric", value_name="Score"
    )

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        data=metrics_melted, x="index", y="Score", hue="Metric", palette="viridis"
    )

    plt.title("Performance Metrics by Class", fontsize=14, pad=20)
    plt.xlabel("Class Label", fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add values on top of bars
    for container in bar_plot.containers:
        bar_plot.bar_label(container, fmt="%.2f", padding=3, fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_performance_metrics.png")
    plt.close()
    print("Generated Class Metrics Plot.")


def plot_confidence_boxplot(df):
    """Boxplot showing confidence ranges for each class, split by correctness."""
    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x="true_label",
        y="confidence",
        hue="correct",
        palette={True: "mediumseagreen", False: "indianred"},
        showfliers=False,
    )

    plt.title("Confidence Levels by Class and Correctness", fontsize=14, pad=20)
    plt.xlabel("Class Label", fontweight="bold")
    plt.ylabel("Confidence Score")
    plt.legend(title="Prediction Status")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confidence_boxplot.png")
    plt.close()
    print("Generated Confidence Boxplot.")


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    if df is not None:
        plot_confusion_matrix(df)
        plot_confidence_distribution(df)
        plot_class_metrics(df)
        plot_confidence_boxplot(df)
        print(f"\nAll plots saved to directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
