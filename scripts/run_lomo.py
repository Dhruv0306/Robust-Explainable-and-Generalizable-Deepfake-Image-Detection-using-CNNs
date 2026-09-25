import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import MODELS, SEEDS
from train import train_model
from evaluate import evaluate_model
from utils import get_device, ensure_gpu_torch_if_needed

MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests" / "lomo"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output" / "lomo"
CHECKPOINT_ROOT = PROJECT_ROOT / "data" / "checkpoints" / "lomo"

HELD_OUT_CATEGORIES = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]


def run_one(held_out, model_name, seed):
    manifest_path = MANIFEST_DIR / f"manifest_holdout_{held_out}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"LOMO manifest not found: {manifest_path}")

    device, device_info = get_device()

    # Keep the device name readable while removing Windows-invalid filename characters.
    device_name = device_info.get("device_name", str(device))
    device_name = re.sub(r'[<>:"/\\|?*]', "_", device_name).strip()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"holdout_{held_out}_{model_name}_{device_name}_{timestamp}_seed{seed}"

    output_dir = OUTPUT_ROOT / run_name
    checkpoint_dir = CHECKPOINT_ROOT / run_name

    # Avoid overwriting or mixing results from a previous run.
    if output_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(
            f"Output already exists for {run_name}. "
            "Move or rename the existing run directory before retrying."
        )

    logging.info(
        "Starting LOMO run: held-out=%s, model=%s, seed=%s, device=%s",
        held_out,
        model_name,
        seed,
        device_name,
    )

    train_result = train_model(
        model_name=model_name,
        manifest_path=manifest_path,
        output_dir=output_dir,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    checkpoint_path = output_dir / "best_checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Training did not produce the expected checkpoint: {checkpoint_path}"
        )

    test_result = evaluate_model(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        split="test",
        device=device,
    )

    logging.info(
        "Completed %s: best_epoch=%s, best_val_loss=%.4f, test_mean_f1=%.4f",
        run_name,
        train_result["best_epoch"],
        train_result["best_val_loss"],
        test_result["video_metrics"]["mean"]["f1"],
    )


def main():
    parser = argparse.ArgumentParser(description="Run one LOMO experiment.")
    parser.add_argument(
        "--holdout",
        required=True,
        choices=HELD_OUT_CATEGORIES,
        help="Manipulation category to leave out of training and validation.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="CNN architecture.",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=SEEDS,
        help="Random seed.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PROJECT_ROOT / "lomo_smoke_test.log", encoding="utf-8"),
        ],
    )

    ensure_gpu_torch_if_needed()
    run_one(args.holdout, args.model, args.seed)


if __name__ == "__main__":
    main()
