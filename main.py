"""
Disease Prediction & Evaluation Demo
Synthetic example with 200 records reflecting the reference slide's
TP/FN/FP/TN counts. Predictions are simulated; this is not a clinical model.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "patient_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    data = pd.read_csv(DATA_FILE)
    y_true = data["actual_disease"]
    y_pred = data["model_prediction"]

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # labels=[1,0] means rows/columns are ordered Disease, Healthy.
    tp, fn, fp, tn = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])

    metrics = {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }

    print("\\n=== Confusion Matrix ===")
    print("Rows = Actual (Disease, Healthy); columns = Predicted (Disease, Healthy)")
    print(pd.DataFrame(cm, index=["Actual Disease","Actual Healthy"],
                      columns=["Predicted Disease","Predicted Healthy"]))
    print("\\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        if key in ("Accuracy","Precision","Recall","F1-Score"):
            print(f"{key}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"{key}: {value}")

    # Save metrics to CSV
    metric_df = pd.DataFrame([metrics])
    metric_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    # Confusion matrix visualization
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Disease","Healthy"],
                yticklabels=["Disease","Healthy"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # Metric scores visualization
    score_names = ["Accuracy","Precision","Recall","F1-Score"]
    score_values = [metrics[name] for name in score_names]
    plt.figure(figsize=(8,5))
    bars = plt.bar(score_names, score_values)
    plt.ylim(0,1)
    plt.ylabel("Score (0 to 1)")
    plt.title("Model Evaluation Metrics")
    plt.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, score_values):
        plt.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.2%}",
                 ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metric_scores.png", dpi=300)
    plt.close()

    # Actual vs predicted distribution
    counts = pd.DataFrame({
        "Actual": y_true.value_counts().reindex([1,0]).values,
        "Predicted": y_pred.value_counts().reindex([1,0]).values
    }, index=["Disease","Healthy"])
    counts.plot(kind="bar", figsize=(8,5))
    plt.title("Actual vs Predicted Class Distribution")
    plt.ylabel("Number of patients")
    plt.xlabel("Class")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300)
    plt.close()

    print(f"\\nVisualizations and metrics saved in: {OUTPUT_DIR}")
    print("\\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Healthy","Disease"],
                               zero_division=0))

if __name__ == "__main__":
    main()
