"""
Approach 2 Robustness Evaluation Runner.

Coordinates end-to-end execution of robustness experiments:
- Loads frozen robustness configuration and freezes snapshots (JSON & TXT).
- Discovers Approach 1 baseline checkpoints.
- Reuses cached clean baseline (or computes once per checkpoint).
- Evaluates enabled image transformations (JPEG, Resize, Darkening, Brightening).
- Retains frame predictions, video aggregations (mean, median, mode), and metrics.
- Computes performance degradation deltas (Delta F1, Delta ROC-AUC, Delta Accuracy).
- Generates visual verification panels (Clean | Sev 1 | Sev 2 | Sev 3).
- Saves machine-readable summary tables and experiment logs.
"""
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from config import (
    MANIFESTS_ROOT,
    OUTPUT_ROOT,
    ROBUSTNESS_CONFIG,
    ROBUSTNESS_ROOT,
    get_robustness_run_name,
)
from dataset import get_robustness_dataloader
from evaluate import (
    evaluate_robustness_condition,
    save_robustness_condition_results,
)
from models import create_model
from robustness_config import (
    get_default_robustness_config,
    get_enabled_conditions,
    save_robustness_config,
    validate_robustness_config,
)
from utils import (
    discover_approach1_checkpoints,
    get_approach1_checkpoint,
    get_device,
    setup_logging,
)
from verify_clean_parity import get_or_compute_clean_baseline
from visual_samples import generate_pilot_visual_examples


def run_robustness_for_checkpoint(
    checkpoint_info: Dict[str, Any],
    manifest_path: Path,
    checkpoint_output_dir: Path,
    config: Dict[str, Any],
    device: torch.device,
    batch_size: int = 64,
    use_amp: bool = False,
    num_workers: int = 4,
) -> pd.DataFrame:
    """
    Execute all enabled robustness conditions for a single checkpoint.

    Returns:
        Summary DataFrame of evaluated conditions and performance degradation deltas.
    """
    model_name = checkpoint_info["model"]
    seed = checkpoint_info["seed"]
    ckpt_path = checkpoint_info["checkpoint_path"]
    checkpoint_output_dir = Path(checkpoint_output_dir)
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"=== Starting robustness evaluation: {model_name} (seed {seed}) ===")
    logging.info(f"Output directory: {checkpoint_output_dir}")

    # 1. Clean Reference Baseline (computed once and cached)
    clean_cache_dir = checkpoint_output_dir / "clean"
    clean_frame_df, clean_video_dfs, clean_metrics = get_or_compute_clean_baseline(
        checkpoint_info=checkpoint_info,
        manifest_path=manifest_path,
        clean_cache_dir=clean_cache_dir,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
        num_workers=num_workers,
    )

    clean_mean_metrics = clean_metrics["video_metrics"]["mean"]
    clean_f1 = clean_mean_metrics["f1"]
    clean_auc = clean_mean_metrics["roc_auc"]
    clean_acc = clean_mean_metrics["accuracy"]

    # 2. Load Model Once for Transformed Conditions
    model = create_model(model_name, pretrained=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    conditions = get_enabled_conditions(config)
    summary_rows = []

    # Record clean row first
    summary_rows.append({
        "model": model_name,
        "seed": seed,
        "condition_id": "clean",
        "transformation": "clean",
        "severity": 0,
        "direction": "none",
        "parameter_name": "none",
        "parameter_value": None,
        "accuracy": clean_acc,
        "precision": clean_mean_metrics["precision"],
        "recall": clean_mean_metrics["recall"],
        "f1": clean_f1,
        "roc_auc": clean_auc,
        "delta_accuracy": 0.0,
        "delta_f1": 0.0,
        "delta_roc_auc": 0.0,
    })

    # 3. Evaluate Transformed Conditions
    for cond in conditions:
        sev = cond["severity"]
        if sev == 0:
            continue  # Clean reference already recorded

        cond_id = cond["condition_id"]
        t_name = cond["transformation"]
        sev_dir_name = f"sev{sev}"
        condition_dir = checkpoint_output_dir / t_name / sev_dir_name

        logging.info(f"Evaluating [{cond_id}] for {model_name} seed {seed}...")

        # Skip if already evaluated
        if (condition_dir / "metrics.json").exists() and (condition_dir / "frame_predictions.csv").exists():
            logging.info(f"Condition {cond_id} already evaluated; loading cached metrics.")
            with open(condition_dir / "metrics.json", "r", encoding="utf-8") as f:
                c_metrics = json.load(f)
        else:
            dataloader = get_robustness_dataloader(
                manifest_path=manifest_path,
                model_name=model_name,
                split="test",
                condition=cond,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            frame_df, video_dfs, c_metrics = evaluate_robustness_condition(
                model=model,
                dataloader=dataloader,
                device=device,
                condition=cond,
                use_amp=use_amp,
            )

            save_robustness_condition_results(
                output_dir=condition_dir,
                frame_df=frame_df,
                video_dfs=video_dfs,
                metrics=c_metrics,
            )

        mean_metrics = c_metrics["video_metrics"]["mean"]
        t_acc = mean_metrics["accuracy"]
        t_f1 = mean_metrics["f1"]
        t_auc = mean_metrics["roc_auc"]

        summary_rows.append({
            "model": model_name,
            "seed": seed,
            "condition_id": cond_id,
            "transformation": t_name,
            "severity": sev,
            "direction": cond["direction"],
            "parameter_name": cond["parameter_name"],
            "parameter_value": cond["parameter_value"],
            "accuracy": t_acc,
            "precision": mean_metrics["precision"],
            "recall": mean_metrics["recall"],
            "f1": t_f1,
            "roc_auc": t_auc,
            "delta_accuracy": t_acc - clean_acc,
            "delta_f1": t_f1 - clean_f1,
            "delta_roc_auc": t_auc - clean_auc,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(checkpoint_output_dir / "condition_summary.csv", index=False)
    return summary_df


def run_robustness_experiment(
    experiment_id: Optional[str] = None,
    output_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    checkpoints: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    batch_size: int = 64,
    use_amp: bool = False,
    num_workers: int = 4,
    save_visuals: bool = True,
) -> Path:
    """
    Run full or pilot robustness experiment.

    Args:
        experiment_id: Unique experiment folder name.
        output_root: Root directory for robustness outputs.
        manifest_path: Path to manifest.csv.
        checkpoints: List of checkpoint dicts to evaluate.
        config: Robustness configuration dict.
        batch_size: Inference batch size.
        use_amp: Enable mixed precision inference.
        num_workers: DataLoader worker count.
        save_visuals: Whether to generate visual example verification strips.

    Returns:
        Path to experiment output directory.
    """
    if experiment_id is None:
        experiment_id = get_robustness_run_name()

    if output_root is None:
        output_root = ROBUSTNESS_ROOT

    if manifest_path is None:
        manifest_path = MANIFESTS_ROOT / "manifest.csv"

    if config is None:
        config = get_default_robustness_config()

    validate_robustness_config(config)

    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 1. Freeze and serialize configuration snapshot
    save_robustness_config(
        config=config,
        output_dir=experiment_dir,
        experiment_id=experiment_id,
        extra_metadata={"timestamp": datetime.now().isoformat()},
    )

    device, dev_info = get_device()
    logging.info(f"Experiment: {experiment_id}")
    logging.info(f"Device: {device} ({dev_info['device_name']})")
    logging.info(f"Target Directory: {experiment_dir}")

    # 2. Checkpoint Selection
    if checkpoints is None or len(checkpoints) == 0:
        checkpoints = discover_approach1_checkpoints()

    logging.info(f"Checkpoints to evaluate: {len(checkpoints)}")

    all_summaries = []
    t0 = time.time()

    for ckpt in checkpoints:
        ckpt_dir_name = f"{ckpt['model']}_seed{ckpt['seed']}"
        ckpt_out_dir = experiment_dir / ckpt_dir_name
        df = run_robustness_for_checkpoint(
            checkpoint_info=ckpt,
            manifest_path=manifest_path,
            checkpoint_output_dir=ckpt_out_dir,
            config=config,
            device=device,
            batch_size=batch_size,
            use_amp=use_amp,
            num_workers=num_workers,
        )
        all_summaries.append(df)

    # Aggregate master summary
    master_df = pd.concat(all_summaries, ignore_index=True)
    master_df.to_csv(experiment_dir / "master_summary.csv", index=False)

    # 3. Visual Example Verification Panels (Plan Section 41 & 42)
    if save_visuals:
        fig_dir = experiment_dir / "figures" / "visual_examples"
        logging.info("Generating candidate visual verification panels...")
        visual_files = generate_pilot_visual_examples(
            manifest_path=manifest_path,
            output_dir=fig_dir,
            config=config,
            max_samples=5,
        )
        logging.info(f"Saved {len(visual_files)} visual example strips to {fig_dir}")

    elapsed = time.time() - t0
    logging.info(f"Robustness experiment {experiment_id} completed in {elapsed:.1f}s")
    return experiment_dir


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Approach 2: Robustness Evaluation Runner")
    parser.add_argument("--pilot", action="store_true", help="Run pilot validation on ResNet50 seed 42 only")
    parser.add_argument("--model", type=str, default=None, help="Filter to specific model architecture")
    parser.add_argument("--seed", type=int, default=None, help="Filter to specific seed")
    parser.add_argument("--experiment-id", type=str, default=None, help="Custom experiment directory identifier")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--no-visuals", action="store_true", help="Skip generating visual example images")

    args = parser.parse_args()

    if args.pilot:
        logging.info("Executing Phase 6 Pilot Robustness Experiment (ResNet50 Seed 42)...")
        pilot_ckpt = get_approach1_checkpoint("resnet50", 42)
        checkpoints = [pilot_ckpt]
        exp_id = args.experiment_id or f"pilot_{get_robustness_run_name()}"
    elif args.model or args.seed:
        all_ckpts = discover_approach1_checkpoints()
        checkpoints = [
            c for c in all_ckpts
            if (args.model is None or c["model"] == args.model)
            and (args.seed is None or c["seed"] == args.seed)
        ]
        exp_id = args.experiment_id or get_robustness_run_name()
    else:
        checkpoints = discover_approach1_checkpoints()
        exp_id = args.experiment_id or get_robustness_run_name()

    run_robustness_experiment(
        experiment_id=exp_id,
        checkpoints=checkpoints,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_visuals=not args.no_visuals,
    )


if __name__ == "__main__":
    main()
