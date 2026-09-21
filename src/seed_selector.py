"""
Approach 3: Automatic Best-Seed and Checkpoint Selector.
Inspects Approach 1 evaluation artifacts to select the best-performing seed
for each architecture using highest video-level F1 (with ROC-AUC tie-breaker).
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


def discover_approach1_runs(output_dir: Path, checkpoint_dir: Path) -> List[Dict[str, Any]]:
    """
    Discover all completed Approach 1 runs with test_results.json and best_checkpoint.pth.

    Args:
        output_dir: Path to data/output/
        checkpoint_dir: Path to data/checkpoints/

    Returns:
        List of dicts with run metadata, metrics, and checkpoint paths
    """
    runs = []
    if not output_dir.exists():
        logging.warning(f"Output directory does not exist: {output_dir}")
        return runs

    for run_path in output_dir.iterdir():
        if not run_path.is_dir():
            continue

        config_file = run_path / "config.json"
        results_file = run_path / "test_results.json"

        if not (config_file.exists() and results_file.exists()):
            continue

        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
            with open(results_file, "r") as f:
                res = json.load(f)

            model_name = cfg.get("model")
            seed = cfg.get("seed")
            run_name = cfg.get("run_name", run_path.name)

            # Extract video-level mean F1 and ROC-AUC
            video_metrics = res.get("video_metrics", {}).get("mean", {})
            video_f1 = video_metrics.get("f1")
            video_auc = video_metrics.get("roc_auc")
            video_acc = video_metrics.get("accuracy")

            if video_f1 is None:
                continue

            # Checkpoint location
            ckpt_path = checkpoint_dir / run_name / "best_checkpoint.pth"

            runs.append({
                "model": model_name,
                "seed": seed,
                "run_name": run_name,
                "run_dir": str(run_path),
                "checkpoint_path": str(ckpt_path),
                "checkpoint_exists": ckpt_path.exists(),
                "video_f1": float(video_f1),
                "video_auc": float(video_auc) if video_auc is not None else 0.0,
                "video_acc": float(video_acc) if video_acc is not None else 0.0,
                "results_file": str(results_file),
            })
        except Exception as e:
            logging.warning(f"Failed to read run at {run_path}: {e}")

    return runs


def select_best_seed(
    model_name: str,
    output_dir: Path,
    checkpoint_dir: Path,
    target_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Select the best-performing seed for a given model architecture.

    Selection rule:
    1. If target_seed is provided, select that seed.
    2. Otherwise: highest video-level F1; if tied, highest video-level ROC-AUC.

    Args:
        model_name: 'xception', 'efficientnet_b0', or 'resnet50'
        output_dir: Path to data/output/
        checkpoint_dir: Path to data/checkpoints/
        target_seed: Optional manual seed override

    Returns:
        Dict with selected run metadata and verified checkpoint path

    Raises:
        ValueError: If no valid runs found or requested seed does not exist
        FileNotFoundError: If the selected checkpoint file is missing on disk
    """
    runs = discover_approach1_runs(output_dir, checkpoint_dir)
    model_runs = [r for r in runs if r["model"] == model_name]

    if not model_runs:
        raise ValueError(f"No Approach 1 runs found for model '{model_name}' in {output_dir}")

    if target_seed is not None:
        matched = [r for r in model_runs if r["seed"] == target_seed]
        if not matched:
            available = [r["seed"] for r in model_runs]
            raise ValueError(f"Seed {target_seed} not found for model '{model_name}'. Available: {available}")
        selected = matched[0]
        selected["selection_mode"] = "manual_override"
    else:
        # Sort descending by video_f1, then video_auc
        model_runs.sort(key=lambda r: (r["video_f1"], r["video_auc"]), reverse=True)
        selected = model_runs[0]
        selected["selection_mode"] = "automatic_best"

    # Validate checkpoint file exists
    ckpt_path = Path(selected["checkpoint_path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file missing for selected run: {ckpt_path}. "
            f"Run: {selected['run_name']}, model: {model_name}, seed: {selected['seed']}"
        )

    logging.info(
        f"Selected {model_name} seed={selected['seed']} ({selected['selection_mode']}): "
        f"video_F1={selected['video_f1']:.4f}, video_AUC={selected['video_auc']:.4f} -> {ckpt_path}"
    )
    return selected


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import OUTPUT_ROOT, CHECKPOINT_ROOT

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    for model in ["xception", "efficientnet_b0", "resnet50"]:
        best = select_best_seed(model, OUTPUT_ROOT, CHECKPOINT_ROOT)
        print(f"[{model.upper()}] Best seed: {best['seed']} (F1={best['video_f1']:.4f}, AUC={best['video_auc']:.4f})")
        print(f"   Checkpoint: {best['checkpoint_path']}")
