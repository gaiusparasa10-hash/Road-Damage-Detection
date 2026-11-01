"""
=============================================================================
Project: AI-Based Road Damage Detection Using Deep Learning (CNN)
File: evaluate_model.py
Description:
    Evaluates the trained CNN model on the validation dataset.
    Calculates key performance metrics:
    - Training & Validation Accuracy
    - Training & Validation Loss
    - Precision
    - Recall
    - F1-Score
    
    Generates and saves the following plots in results/:
    - results/accuracy_plot.png
    - results/loss_plot.png
    - results/confusion_matrix.png
    
    Also exports actual evaluation metrics to results/evaluation_metrics.json
    for dynamic display in the Streamlit frontend.
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# ---------------------------------------------------------------------------
# PATH DEFINITIONS & CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "road_damage_model.keras")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

ACCURACY_PLOT_PATH = os.path.join(RESULTS_DIR, "accuracy_plot.png")
LOSS_PLOT_PATH = os.path.join(RESULTS_DIR, "loss_plot.png")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
HISTORY_JSON_PATH = os.path.join(RESULTS_DIR, "history.json")
CLASSES_JSON_PATH = os.path.join(RESULTS_DIR, "class_names.json")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
RANDOM_SEED = 42


def plot_accuracy_and_loss(history_data):
    """
    Plots and saves separate training vs validation curves for Accuracy and Loss.
    """
    epochs = range(1, len(history_data["accuracy"]) + 1)

    # 1. Plot Training vs Validation Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [a * 100 for a in history_data["accuracy"]], "bo-", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, [va * 100 for va in history_data["val_accuracy"]], "rs-", label="Validation Accuracy", linewidth=2)
    plt.title("Training vs Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH, dpi=300)
    plt.close()
    print(f"[SUCCESS] Accuracy plot saved to: {ACCURACY_PLOT_PATH}")

    # 2. Plot Training vs Validation Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_data["loss"], "bo-", label="Training Loss", linewidth=2)
    plt.plot(epochs, history_data["val_loss"], "rs-", label="Validation Loss", linewidth=2)
    plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Binary Crossentropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=300)
    plt.close()
    print(f"[SUCCESS] Loss plot saved to: {LOSS_PLOT_PATH}")


def plot_confusion_matrix_heatmap(cm, class_display_names):
    """
    Plots a clean, professional confusion matrix with annotations.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    # Add colorbar
    fig.colorbar(cax)

    # Configure axis labels
    ax.set_xticks(range(len(class_display_names)))
    ax.set_yticks(range(len(class_display_names)))
    ax.set_xticklabels(class_display_names, fontsize=11)
    ax.set_yticklabels(class_display_names, fontsize=11)

    # Move ticks to bottom
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

    # Annotate numbers inside each matrix cell
    total_samples = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_value = cm[i, j]
            pct = (cell_value / total_samples) * 100 if total_samples > 0 else 0
            text_color = "white" if cell_value > (cm.max() / 2) else "black"
            ax.text(
                j, i, f"{cell_value}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color, fontsize=12, fontweight="bold"
            )

    plt.title("Confusion Matrix", fontsize=14, fontweight="bold", pad=16)
    plt.xlabel("Predicted Road Condition", fontsize=12, labelpad=10)
    plt.ylabel("Actual Road Condition", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"[SUCCESS] Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


def evaluate():
    """
    Main evaluation pipeline.
    """
    print("=" * 70)
    print(" AI-BASED ROAD DAMAGE DETECTION - MODEL EVALUATION")
    print("=" * 70)

    # 1. Verify model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Trained model file not found at: {MODEL_PATH}")
        print("Please train the model first by running: python train_model.py")
        return

    # 2. Verify processed dataset exists
    if not os.path.exists(PROCESSED_DIR):
        print(f"[ERROR] Processed data directory not found at: {PROCESSED_DIR}")
        print("Please run preprocess.py first.")
        return

    # 3. Load trained model
    print(f"[INFO] Loading trained CNN model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 4. Load validation dataset
    print(f"[INFO] Loading validation dataset (20% split, Seed: {RANDOM_SEED})...")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR,
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    class_names = val_dataset.class_names
    display_names = ["Damaged Road" if c == "Damaged" else "Normal Road" for c in class_names]
    print(f"[INFO] Classes: {class_names} -> Display: {display_names}")

    # 5. Extract actual labels and run batch predictions
    all_true_labels = []
    all_predicted_probs = []

    for images, labels in val_dataset:
        batch_preds = model.predict(images, verbose=0)
        all_true_labels.extend(labels.numpy().flatten())
        all_predicted_probs.extend(batch_preds.flatten())

    all_true_labels = np.array(all_true_labels, dtype=int)
    all_predicted_probs = np.array(all_predicted_probs)
    all_pred_labels = (all_predicted_probs >= 0.5).astype(int)

    # 6. Compute Metrics
    # In alphabetical order: index 0 = Damaged, index 1 = Normal
    val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, average="weighted", zero_division=0
    )

    # Check for saved training history
    train_loss = None
    train_acc = None
    if os.path.exists(HISTORY_JSON_PATH):
        with open(HISTORY_JSON_PATH, "r") as f:
            history_data = json.load(f)
            if history_data.get("accuracy"):
                train_acc = history_data["accuracy"][-1]
            if history_data.get("loss"):
                train_loss = history_data["loss"][-1]
            plot_accuracy_and_loss(history_data)

    # 7. Generate and save Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    plot_confusion_matrix_heatmap(cm, display_names)

    # 8. Save Metrics to JSON
    metrics_summary = {
        "training_accuracy": round(float(train_acc) * 100, 2) if train_acc is not None else None,
        "training_loss": round(float(train_loss), 4) if train_loss is not None else None,
        "validation_accuracy": round(float(val_acc) * 100, 2),
        "validation_loss": round(float(val_loss), 4),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "total_validation_samples": int(len(all_true_labels))
    }

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[SUCCESS] Metrics summary saved to: {METRICS_JSON_PATH}")

    # 9. Terminal Report
    print("\n" + "=" * 70)
    print(" ACTUAL MODEL EVALUATION REPORT")
    print("=" * 70)
    if metrics_summary["training_accuracy"] is not None:
        print(f"  Training Accuracy   : {metrics_summary['training_accuracy']:.2f}%")
        print(f"  Training Loss       : {metrics_summary['training_loss']:.4f}")
    print(f"  Validation Accuracy : {metrics_summary['validation_accuracy']:.2f}%")
    print(f"  Validation Loss     : {metrics_summary['validation_loss']:.4f}")
    print(f"  Weighted Precision  : {metrics_summary['precision']:.2f}%")
    print(f"  Weighted Recall     : {metrics_summary['recall']:.2f}%")
    print(f"  Weighted F1-Score   : {metrics_summary['f1_score']:.2f}%")
    print(f"  Validation Samples  : {metrics_summary['total_validation_samples']}")
    print("-" * 70)
    print("\nDetailed Per-Class Classification Report:")
    print(classification_report(all_true_labels, all_pred_labels, target_names=display_names, zero_division=0))
    print("=" * 70)
    print("Plots generated in results/:")
    print("  - results/accuracy_plot.png")
    print("  - results/loss_plot.png")
    print("  - results/confusion_matrix.png")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
