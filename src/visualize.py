"""
Visualization utilities: training curves, confusion matrix, ROC curve.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import json


def plot_training_curves(history: dict, output_path: Path):
    """Plot training and validation loss/accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker='s')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker='o')
    ax2.plot(epochs, history["val_acc"], label="Val Acc", marker='s')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: list, output_path: Path, labels=["Real", "Fake"]):
    """Plot confusion matrix heatmap"""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}


def save_all_visualizations(history_path: Path, results_path: Path, output_dir: Path):
    """Generate all visualizations from saved results"""
    # Training curves
    with open(history_path) as f:
        history = json.load(f)
    plot_training_curves(history, output_dir / "training_curves.png")

    # Test results
    with open(results_path) as f:
        results = json.load(f)

    # Confusion matrix (frame-level)
    plot_confusion_matrix(
        results["frame_metrics"]["confusion_matrix"],
        output_dir / "confusion_matrix_frame.png"
    )

    # Confusion matrix (video-level, primary aggregation)
    primary_agg = "mean"  # from config.PRIMARY_AGGREGATION
    plot_confusion_matrix(
        results["video_metrics"][primary_agg]["confusion_matrix"],
        output_dir / f"confusion_matrix_video_{primary_agg}.png"
    )

    # ROC curves would need predictions CSV - skip for now or load from CSV


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    history_path = output_dir / "history.json"
    results_path = output_dir / "test_results.json"

    if history_path.exists() and results_path.exists():
        save_all_visualizations(history_path, results_path, output_dir)
        print(f"Visualizations saved to {output_dir}")
    else:
        print("Missing history.json or test_results.json")
