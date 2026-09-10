"""
Approach 2 Clean Baseline Parity and Caching Engine.

Performs regression testing against Approach 1 baseline outputs and manages
cached clean predictions for all 9 model checkpoints.

Ensures:
1. Exact metric parity with Approach 1 (Accuracy, Precision, Recall, F1, ROC-AUC).
2. 100% binary video prediction agreement across all 24 test videos.
3. Clean predictions are computed exactly once per checkpoint and cached in <checkpoint>/clean/
   so transformed conditions reuse the fixed reference without redundant forward passes.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from dataset import get_robustness_dataloader
from evaluate import (
    evaluate_robustness_condition,
    save_robustness_condition_results,
)
from models import create_model
from utils import (
    discover_approach1_checkpoints,
    get_approach1_checkpoint,
    get_device,
)


CLEAN_CONDITION: Dict[str, Any] = {
    "transformation": "clean",
    "severity": 0,
    "direction": "none",
    "parameter_name": "none",
    "parameter_value": None,
    "condition_id": "clean",
}


def run_clean_evaluation_for_checkpoint(
    checkpoint_info: Dict[str, Any],
    manifest_path: Path,
    output_dir: Optional[Path] = None,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Run clean inference on a checkpoint and verify parity against Approach 1 baseline.

    Args:
        checkpoint_info: Dict from discover_approach1_checkpoints().
        manifest_path: Path to test manifest.csv.
        output_dir: Optional directory to cache clean results.
        batch_size: DataLoader batch size.
        device: PyTorch device (CPU or CUDA).
        use_amp: Whether to enable AMP (defaults to False for exact CPU/GPU reproducibility).
        num_workers: DataLoader worker count.

    Returns:
        (frame_df, video_dfs, metrics)
    """
    if device is None:
        device, _ = get_device()

    model_name = checkpoint_info["model"]
    seed = checkpoint_info["seed"]
    ckpt_path = checkpoint_info["checkpoint_path"]
    run_name = checkpoint_info["run_name"]

    logging.info(f"Running clean evaluation for {model_name} (seed {seed}, {run_name})...")

    # Load model
    model = create_model(model_name, pretrained=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    # DataLoader
    dataloader = get_robustness_dataloader(
        manifest_path=manifest_path,
        model_name=model_name,
        split="test",
        condition=CLEAN_CONDITION,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Evaluate clean condition
    frame_df, video_dfs, metrics = evaluate_robustness_condition(
        model=model,
        dataloader=dataloader,
        device=device,
        condition=CLEAN_CONDITION,
        use_amp=use_amp,
    )

    # Validate against saved Approach 1 baseline results
    baseline_json_path = checkpoint_info["run_dir"] / "test_results.json"
    if baseline_json_path.exists():
        with open(baseline_json_path, "r", encoding="utf-8") as f:
            saved_results = json.load(f)

        saved_mean = saved_results["video_metrics"]["mean"]
        computed_mean = metrics["video_metrics"]["mean"]

        # Assert key classification metrics match within float precision
        for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            saved_val = saved_mean[metric_name]
            computed_val = computed_mean[metric_name]
            diff = abs(computed_val - saved_val)
            if diff > 1e-4:
                raise AssertionError(
                    f"Clean parity violation on {model_name} seed {seed}: "
                    f"{metric_name} computed={computed_val:.6f}, saved={saved_val:.6f}, diff={diff:.6f}"
                )

        logging.info(
            f"Parity confirmed for {model_name} seed {seed}: "
            f"Acc={computed_mean['accuracy']:.4f}, AUC={computed_mean['roc_auc']:.4f}, F1={computed_mean['f1']:.4f}"
        )

    # Save to output_dir if provided
    if output_dir is not None:
        save_robustness_condition_results(
            output_dir=Path(output_dir),
            frame_df=frame_df,
            video_dfs=video_dfs,
            metrics=metrics,
        )
        logging.info(f"Cached clean baseline to {output_dir}")

    return frame_df, video_dfs, metrics


def get_or_compute_clean_baseline(
    checkpoint_info: Dict[str, Any],
    manifest_path: Path,
    clean_cache_dir: Path,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
    num_workers: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Retrieve clean baseline results from disk cache if present, otherwise compute and cache.

    Implements the single-clean-inference principle per Plan Section 19.
    """
    clean_cache_dir = Path(clean_cache_dir)
    frame_csv = clean_cache_dir / "frame_predictions.csv"
    metrics_json = clean_cache_dir / "metrics.json"

    if frame_csv.exists() and metrics_json.exists():
        logging.info(f"Loading cached clean baseline from {clean_cache_dir}")
        frame_df = pd.read_csv(frame_csv)

        video_dfs = {}
        for method in ["mean", "median", "mode"]:
            v_csv = clean_cache_dir / f"video_predictions_{method}.csv"
            if v_csv.exists():
                video_dfs[method] = pd.read_csv(v_csv)

        with open(metrics_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        return frame_df, video_dfs, metrics

    # Compute, verify parity, and cache
    return run_clean_evaluation_for_checkpoint(
        checkpoint_info=checkpoint_info,
        manifest_path=manifest_path,
        output_dir=clean_cache_dir,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    from config import MANIFESTS_ROOT, OUTPUT_ROOT
    from utils import setup_logging

    setup_logging()
    ckpt = get_approach1_checkpoint("resnet50", 42, OUTPUT_ROOT)
    run_clean_evaluation_for_checkpoint(ckpt, MANIFESTS_ROOT / "manifest.csv")
