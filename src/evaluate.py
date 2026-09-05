"""
Evaluation: frame-level inference, video-level aggregation, metrics.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import sys
sys.path.append(str(Path(__file__).parent))
from config import *
from models import create_model, get_model_input_size
from dataset import get_dataloader
from utils import get_device, get_batch_size


@torch.no_grad()
def get_frame_predictions(model, dataloader, device, use_amp=True) -> pd.DataFrame:
    """
    Run inference on all frames, return DataFrame with predictions.
    """
    model.eval()

    frame_results = []

    for images, labels, video_ids in tqdm(dataloader, desc="Inference"):
        images = images.to(device)

        with autocast(enabled=use_amp):
            logits = model(images).squeeze(1)  # (B,)

        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.numpy()

        for i in range(len(video_ids)):
            frame_results.append({
                "video_id": video_ids[i],
                "label": int(labels_np[i]),
                "prob_fake": float(probs[i]),
                "pred_fake": int(probs[i] >= 0.5),
            })

    return pd.DataFrame(frame_results)


def aggregate_to_video_level(frame_df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """
    Aggregate frame predictions to video level.

    Args:
        frame_df: DataFrame with frame-level predictions
        method: 'mean', 'median', or 'mode'

    Returns:
        DataFrame with video-level predictions
    """
    video_results = []

    for video_id, group in frame_df.groupby("video_id"):
        label = group["label"].iloc[0]  # All frames in video have same label

        if method == "mean":
            prob_fake = group["prob_fake"].mean()
        elif method == "median":
            prob_fake = group["prob_fake"].median()
        elif method == "mode":
            # Mode of binary predictions
            mode_pred = group["pred_fake"].mode()[0]
            prob_fake = float(mode_pred)  # 0.0 or 1.0
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        pred_fake = int(prob_fake >= 0.5)

        video_results.append({
            "video_id": video_id,
            "label": label,
            "prob_fake": prob_fake,
            "pred_fake": pred_fake,
        })

    return pd.DataFrame(video_results)


def compute_metrics(df: pd.DataFrame) -> Dict:
    """Compute classification metrics from predictions DataFrame"""
    y_true = df["label"].values
    y_pred = df["pred_fake"].values
    y_prob = df["prob_fake"].values

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    return metrics


def evaluate_model(
    model_name: str,
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    split: str = "test",
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Full evaluation pipeline.

    Returns dict with frame-level and video-level metrics.
    """
    logging.info(f"Evaluating {model_name} on {split} set")

    # Device
    if device is None:
        device, device_info = get_device()
    else:
        _, device_info = get_device()

    use_amp = device_info["amp_enabled"]

    # Load model
    model = create_model(model_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Data
    batch_size = get_batch_size(device, get_model_input_size(model_name))
    dataloader = get_dataloader(manifest_path, split, model_name, batch_size, NUM_WORKERS, shuffle=False)

    # Frame-level predictions
    frame_df = get_frame_predictions(model, dataloader, device, use_amp)

    # Frame-level metrics
    frame_metrics = compute_metrics(frame_df)
    logging.info(f"Frame-level metrics: Acc={frame_metrics['accuracy']:.4f}, F1={frame_metrics['f1']:.4f}")

    # Video-level aggregation
    video_results = {}
    for method in VIDEO_AGGREGATION_METHODS:
        video_df = aggregate_to_video_level(frame_df, method)
        video_metrics = compute_metrics(video_df)
        video_results[method] = {
            "predictions": video_df,
            "metrics": video_metrics,
        }
        logging.info(f"Video-level ({method}): Acc={video_metrics['accuracy']:.4f}, F1={video_metrics['f1']:.4f}")

    # Per-manipulation breakdown (on primary aggregation method)
    primary_video_df = video_results[PRIMARY_AGGREGATION]["predictions"]
    # Load manifest to get category info
    manifest_df = pd.read_csv(manifest_path)
    manifest_df = manifest_df[manifest_df["split"] == split]
    video_to_category = manifest_df.groupby("video_id")["category"].first().to_dict()
    primary_video_df["category"] = primary_video_df["video_id"].map(video_to_category)

    per_category_metrics = {}
    for category in ["Original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        cat_df = primary_video_df[primary_video_df["category"] == category]
        if len(cat_df) > 0:
            cat_metrics = compute_metrics(cat_df)
            per_category_metrics[category] = cat_metrics
            logging.info(f"{category}: Acc={cat_metrics['accuracy']:.4f}, F1={cat_metrics['f1']:.4f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Frame predictions
    frame_df.to_csv(output_dir / f"{split}_frame_predictions.csv", index=False)

    # Video predictions
    for method in VIDEO_AGGREGATION_METHODS:
        video_results[method]["predictions"].to_csv(
            output_dir / f"{split}_video_predictions_{method}.csv", index=False
        )

    # Metrics
    results = {
        "split": split,
        "frame_metrics": frame_metrics,
        "video_metrics": {m: video_results[m]["metrics"] for m in VIDEO_AGGREGATION_METHODS},
        "per_category_metrics": per_category_metrics,
    }

    with open(output_dir / f"{split}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate visualizations
    try:
        from visualize import plot_confusion_matrix, plot_roc_curve

        # Confusion matrices
        plot_confusion_matrix(
            frame_metrics["confusion_matrix"],
            output_dir / f"{split}_confusion_matrix_frame.png"
        )

        for method in VIDEO_AGGREGATION_METHODS:
            plot_confusion_matrix(
                video_results[method]["metrics"]["confusion_matrix"],
                output_dir / f"{split}_confusion_matrix_video_{method}.png"
            )

        # ROC curve (primary video aggregation)
        primary_video_df = video_results[PRIMARY_AGGREGATION]["predictions"]
        roc_data = plot_roc_curve(
            primary_video_df["label"].values,
            primary_video_df["prob_fake"].values,
            output_dir / f"{split}_roc_curve.png"
        )
        # Save ROC data
        with open(output_dir / f"{split}_roc_data.json", "w") as f:
            json.dump(roc_data, f, indent=2)

        logging.info("Visualizations saved")
    except Exception as e:
        logging.warning(f"Failed to generate visualizations: {e}")

    logging.info(f"Results saved to {output_dir}")

    return results


if __name__ == "__main__":
    from utils import setup_logging, get_device

    setup_logging()

    # Test evaluation
    manifest = MANIFESTS_ROOT / "manifest.csv"
    checkpoint = OUTPUT_ROOT / "test_run" / "best_checkpoint.pth"
    output_dir = OUTPUT_ROOT / "test_run"

    if manifest.exists() and checkpoint.exists():
        evaluate_model("xception", checkpoint, manifest, output_dir, split="test")
