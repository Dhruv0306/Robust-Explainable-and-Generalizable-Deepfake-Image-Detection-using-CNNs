"""
Experiment runner: Full pipeline for Approach 1.
Runs data preprocessing + trains all models with all seeds.
"""
import argparse
import logging
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from config import *
from utils import setup_logging, set_seed, get_device, save_config, get_batch_size
from models import get_model_input_size


def run_data_pipeline():
    """Run full data preprocessing pipeline"""
    logging.info("=" * 80)
    logging.info("STEP 1: Data Preprocessing")
    logging.info("=" * 80)

    # 1. Inspect dataset
    from data.inspect_dataset import inspect_dataset
    video_paths = inspect_dataset()

    # 2. Build relationship graph
    from data.build_relationship_graph import build_relationship_graph, find_connected_components
    graph = build_relationship_graph(video_paths)
    components = find_connected_components(graph)

    # 3. Create splits
    from data.create_splits import assign_splits, save_splits
    splits = assign_splits(components, seed=42)
    save_splits(splits, SPLITS_ROOT)

    # 4. Extract frames
    from data.extract_frames import extract_all_frames
    import json
    frame_metadata = extract_all_frames(video_paths, splits)
    MANIFESTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFESTS_ROOT / "frame_extraction_metadata.json", "w") as f:
        json.dump(frame_metadata, f, indent=2)

    # 5. Detect and crop faces
    from data.detect_and_crop_faces import process_all_videos
    face_metadata = process_all_videos(frame_metadata)
    with open(MANIFESTS_ROOT / "face_processed_metadata.json", "w") as f:
        json.dump(face_metadata, f, indent=2)

    # 6. Build manifest
    from data.build_manifest import build_manifest
    build_manifest(face_metadata)

    logging.info("Data preprocessing complete ✓")


def run_single_experiment(model_name: str, seed: int, manifest_path: Path, device=None):
    """Run single training + evaluation"""
    logging.info("=" * 80)
    logging.info(f"Training: {model_name} | Seed: {seed}")
    logging.info("=" * 80)

    # Run directory
    run_name = get_run_name(model_name)
    output_dir = OUTPUT_ROOT / run_name
    checkpoint_dir = CHECKPOINT_ROOT / run_name

    # Config
    device, device_info = get_device() if device is None else (device, {})
    input_size = get_model_input_size(model_name)
    batch_size = get_batch_size(device, input_size)

    config = {
        "model": model_name,
        "seed": seed,
        "device": str(device),
        "device_name": device_info.get("device_name", "unknown"),
        "cuda_available": device_info.get("cuda_available", False),
        "amp_enabled": device_info.get("amp_enabled", False),
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "input_size": input_size,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "hflip_prob": HFLIP_PROB,
        "gaussian_blur_prob": GAUSSIAN_BLUR_PROB,
        "frame_sampling_interval": FRAME_SAMPLING_INTERVAL,
        "min_usable_frames": MIN_USABLE_FRAMES_PER_VIDEO,
        "bbox_expansion_margin": BBOX_EXPANSION_MARGIN,
        "iou_threshold": IOU_THRESHOLD,
        "run_name": run_name,
    }

    save_config(config, output_dir)
    logging.info(f"Config saved to {output_dir}")

    # Train
    from train import train_model
    train_results = train_model(model_name, manifest_path, output_dir, seed, device)

    # Copy best checkpoint to checkpoint dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(output_dir / "best_checkpoint.pth", checkpoint_dir / "best_checkpoint.pth")
    shutil.copy(output_dir / "config.json", checkpoint_dir / "config.json")
    shutil.copy(output_dir / "config.txt", checkpoint_dir / "config.txt")
    logging.info(f"Checkpoint saved to {checkpoint_dir}")

    # Evaluate on test set
    from evaluate import evaluate_model
    test_results = evaluate_model(
        model_name,
        checkpoint_dir / "best_checkpoint.pth",
        manifest_path,
        output_dir,
        split="test",
        device=device,
    )

    logging.info(f"Experiment complete: {run_name}")
    logging.info(f"Test Accuracy: {test_results['video_metrics'][PRIMARY_AGGREGATION]['accuracy']:.4f}")
    logging.info(f"Test F1: {test_results['video_metrics'][PRIMARY_AGGREGATION]['f1']:.4f}")


def run_experiment_matrix(manifest_path: Path):
    """Run full experiment matrix: 3 models × 3 seeds"""
    logging.info("=" * 80)
    logging.info("STEP 2: Training & Evaluation")
    logging.info("=" * 80)

    device, _ = get_device()

    total_runs = len(MODELS) * len(SEEDS)
    current_run = 0

    for model_name in MODELS.keys():
        for seed in SEEDS:
            current_run += 1
            logging.info(f"\nRun {current_run}/{total_runs}")
            run_single_experiment(model_name, seed, manifest_path, device)


def main():
    parser = argparse.ArgumentParser(description="Approach 1: CNN Baseline Experiment Runner")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()), help="Run single model")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Run single seed")
    args = parser.parse_args()

    # Setup logging
    log_file = PROJECT_ROOT / "experiment.log"
    setup_logging(log_file)

    logging.info("=" * 80)
    logging.info("Approach 1: CNN Baseline for Deepfake Detection")
    logging.info("=" * 80)

    # Data pipeline
    if not args.skip_data:
        run_data_pipeline()
    else:
        logging.info("Skipping data preprocessing (--skip-data)")

    # Check manifest
    manifest_path = MANIFESTS_ROOT / "manifest.csv"
    if not manifest_path.exists():
        logging.error(f"Manifest not found: {manifest_path}")
        logging.error("Run without --skip-data first")
        sys.exit(1)

    # Training
    if args.model and args.seed:
        # Single run
        run_single_experiment(args.model, args.seed, manifest_path)
    elif args.model:
        # Single model, all seeds
        device, _ = get_device()
        for seed in SEEDS:
            run_single_experiment(args.model, seed, manifest_path, device)
    else:
        # Full matrix
        run_experiment_matrix(manifest_path)

    logging.info("=" * 80)
    logging.info("All experiments complete ✓")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
