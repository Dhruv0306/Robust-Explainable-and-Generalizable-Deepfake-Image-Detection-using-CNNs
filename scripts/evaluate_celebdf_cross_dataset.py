from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from evaluate import evaluate_model


MANIFEST = (
    ROOT / "data" / "CelebDF_processed" / "manifests"
    / "manifest_cross_dataset.csv"
)
OUTPUT_ROOT = ROOT / "data" / "output" / "cross_dataset_celebdf"

# Explicitly select the September 24 Approach 1 baseline batch.
RUNS = [
    ("xception", 42, "xception_LAPTOP-KTKP3V55_2026-09-24_17-34-44"),
    ("xception", 123, "xception_LAPTOP-KTKP3V55_2026-09-24_18-48-31"),
    ("xception", 2024, "xception_LAPTOP-KTKP3V55_2026-09-24_19-57-26"),
    ("efficientnet_b0", 42, "efficientnet_b0_LAPTOP-KTKP3V55_2026-09-24_20-55-13"),
    ("efficientnet_b0", 123, "efficientnet_b0_LAPTOP-KTKP3V55_2026-09-24_21-11-57"),
    ("efficientnet_b0", 2024, "efficientnet_b0_LAPTOP-KTKP3V55_2026-09-24_21-48-23"),
    ("resnet50", 42, "resnet50_LAPTOP-KTKP3V55_2026-09-24_22-13-00"),
    ("resnet50", 123, "resnet50_LAPTOP-KTKP3V55_2026-09-24_22-37-13"),
    ("resnet50", 2024, "resnet50_LAPTOP-KTKP3V55_2026-09-24_23-01-16"),
]


def validate_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)

    required = {"frame_path", "video_id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    if set(df["split"].dropna().unique()) != {"test"}:
        raise ValueError("Expected the manifest to contain only the test split.")

    if not set(df["label"].dropna().unique()).issubset({"real", "fake"}):
        raise ValueError("Unexpected label values. Expected 'real' and 'fake'.")

    if df["video_id"].isna().any():
        raise ValueError("Manifest contains empty video_id values.")

    missing_paths = [
        p for p in df["frame_path"].astype(str)
        if not Path(p).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} manifest frame paths are missing. "
            f"Example: {missing_paths[0]}"
        )

    # Confirm each video has only one ground-truth label.
    label_counts = df.groupby("video_id")["label"].nunique()
    conflicting = label_counts[label_counts > 1]
    if not conflicting.empty:
        raise ValueError(
            f"Videos have conflicting labels: {conflicting.index[:5].tolist()}"
        )

    print(f"Manifest validated: {len(df):,} frames, "
          f"{df['video_id'].nunique():,} videos")
    print("Label counts:", df.groupby("label")["video_id"].nunique().to_dict())


def main():
    validate_manifest()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for model_name, seed, run_name in RUNS:
        run_dir = ROOT / "data" / "output" / run_name
        checkpoint = run_dir / "best_checkpoint.pth"

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")

        result_dir = OUTPUT_ROOT / model_name / f"seed_{seed}"
        result_file = result_dir / "test_results.json"

        if result_file.exists():
            raise FileExistsError(
                f"Results already exist for {model_name}, seed {seed}: "
                f"{result_file}\nMove or rename the previous result folder "
                "before rerunning to avoid accidental overwrites."
            )

        print(f"\n=== Evaluating {model_name}, seed {seed} ===")
        print(f"Checkpoint: {checkpoint}")
        print(f"Output: {result_dir}")

        results = evaluate_model(
            model_name=model_name,
            checkpoint_path=checkpoint,
            manifest_path=MANIFEST,
            output_dir=result_dir,
            split="test",
        )

        primary = results["video_metrics"]["mean"]
        print(
            f"Video-level mean aggregation: "
            f"accuracy={primary['accuracy']:.4f}, "
            f"precision={primary['precision']:.4f}, "
            f"recall={primary['recall']:.4f}, "
            f"F1={primary['f1']:.4f}, "
            f"ROC-AUC={primary['roc_auc']:.4f}"
        )

    print("\nAll nine evaluations completed.")
    print(f"Results saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()