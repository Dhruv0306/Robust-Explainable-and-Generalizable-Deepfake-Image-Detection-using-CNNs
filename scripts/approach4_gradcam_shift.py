"""
Approach 4: Generate Grad-CAM examples for seen vs. held-out manipulations.

Seen examples: fake frames in the validation split from manipulation
categories other than the held-out category.
Unseen examples: fake frames in the test split from the held-out category.

Both sets are explained with the same explicitly supplied LOO checkpoint.
Sampling is deterministic at the video level.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from explainability_gradcam import GradCAMGenerator
from utils import get_device


HELD_OUT_CATEGORIES = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]


def safe_name(value):
    return re.sub(r'[<>:"/\\|?*]', "_", str(value)).strip(" ._")


def select_examples(
    manifest_path,
    heldout,
    videos_per_category,
    frames_per_video,
    sample_seed,
):
    df = pd.read_csv(manifest_path)

    required = {
        "frame_path", "video_id", "category", "label",
        "split", "original_frame_number",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    # Restrict to fake frames. The held-out category must only be taken
    # from test; seen categories must only be taken from validation.
    fake = df[df["label"].astype(str).str.lower() == "fake"].copy()
    fake["video_id"] = fake["video_id"].astype(str)

    seen = fake[
        (fake["split"] == "val") &
        (fake["category"] != heldout)
    ].copy()
    unseen = fake[
        (fake["split"] == "test") &
        (fake["category"] == heldout)
    ].copy()

    selected = []

    for condition, subset in [("seen", seen), ("unseen", unseen)]:
        categories = (
            [c for c in sorted(subset["category"].unique()) if c != heldout]
            if condition == "seen"
            else [heldout]
        )

        if not categories:
            raise ValueError(f"No {condition} fake categories found.")

        for category in categories:
            category_df = subset[subset["category"] == category]
            video_ids = sorted(category_df["video_id"].unique())

            if not video_ids:
                raise ValueError(
                    f"No videos for condition={condition}, category={category}"
                )

            rng = np.random.default_rng(
                sample_seed + sum(ord(ch) for ch in f"{condition}:{category}")
            )
            count = min(videos_per_category, len(video_ids))
            chosen_videos = sorted(
                rng.choice(video_ids, size=count, replace=False).tolist()
            )

            for video_id in chosen_videos:
                video_df = category_df[
                    category_df["video_id"] == video_id
                ].sort_values("original_frame_number")

                frame_count = min(frames_per_video, len(video_df))
                if frame_count == 0:
                    continue

                # Use a separate deterministic RNG for frame selection.
                frame_rng = np.random.default_rng(
                    sample_seed + sum(
                        ord(ch) for ch in f"{condition}:{category}:{video_id}"
                    )
                )
                chosen_indices = sorted(
                    frame_rng.choice(
                        video_df.index.to_numpy(),
                        size=frame_count,
                        replace=False,
                    ).tolist()
                )
                selected.append(video_df.loc[chosen_indices])

    if not selected:
        raise ValueError("Sampling produced no examples.")

    return pd.concat(selected, ignore_index=True)


def save_cam_visualization(image_rgb, cam, output_path, alpha=0.40):
    """Save original, heatmap, and overlay in one horizontal image."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(
        np.uint8(np.clip(cam, 0.0, 1.0) * 255),
        cv2.COLORMAP_JET,
    )
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0)

    # Label each panel for presentation use.
    panels = []
    for panel, label in [
        (image_bgr, "Original"),
        (heatmap, "Grad-CAM"),
        (overlay, "Overlay"),
    ]:
        panel = panel.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(
            panel, label, (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        panels.append(panel)

    combined = cv2.hconcat(panels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), combined):
        raise IOError(f"Could not write visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate seen/unseen Grad-CAM examples for one LOO run."
    )
    parser.add_argument("--holdout", required=True, choices=HELD_OUT_CATEGORIES)
    parser.add_argument(
        "--model", required=True,
        choices=["xception", "efficientnet_b0", "resnet50"],
    )
    parser.add_argument("--seed", required=True, type=int,
                        help="Training seed of the selected LOO checkpoint.")
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to this run's best_checkpoint.pth.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Defaults to the matching LOO holdout manifest.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Defaults to data/output/approach4_gradcam_shift/<run-id>.",
    )
    parser.add_argument("--videos-per-category", type=int, default=12)
    parser.add_argument("--frames-per-video", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    if args.videos_per_category < 1 or args.frames_per_video < 1:
        parser.error("--videos-per-category and --frames-per-video must be >= 1")

    manifest = args.manifest or (
        PROJECT_ROOT / "data" / "manifests" / "lomo" /
        f"manifest_holdout_{args.holdout}.csv"
    )
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    run_id = (
        f"holdout_{args.holdout}_{args.model}_seed{args.seed}"
    )
    output_dir = args.output or (
        PROJECT_ROOT / "data" / "output" /
        "approach4_gradcam_shift" / run_id
    )
    image_dir = output_dir / "visualizations"
    cam_dir = output_dir / "cams"

    examples = select_examples(
        manifest_path=manifest,
        heldout=args.holdout,
        videos_per_category=args.videos_per_category,
        frames_per_video=args.frames_per_video,
        sample_seed=args.sample_seed,
    )

    device, device_info = get_device()
    logging.info("Device: %s (%s)", device, device_info.get("device_name"))

    generator = GradCAMGenerator(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    records = []
    try:
        for index, row in examples.iterrows():
            condition = (
                "unseen" if row["category"] == args.holdout else "seen"
            )
            category = str(row["category"])
            video_id = safe_name(row["video_id"])
            frame_number = int(row["original_frame_number"])

            image_path = Path(str(row["frame_path"]))
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read frame: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            probability, prediction = generator.predict(image_rgb)
            cam = generator.generate_cam(
                image_rgb,
                target_shape=image_rgb.shape[:2],
            )

            sample_id = (
                f"{index:05d}_{condition}_{safe_name(category)}_"
                f"{video_id}_frame{frame_number:04d}"
            )
            vis_path = image_dir / f"{sample_id}.jpg"
            raw_cam_path = cam_dir / f"{sample_id}.npy"

            save_cam_visualization(image_rgb, cam, vis_path)
            raw_cam_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(raw_cam_path, cam)

            records.append({
                "sample_id": sample_id,
                "condition": condition,
                "category": category,
                "split": str(row["split"]),
                "video_id": str(row["video_id"]),
                "original_frame_number": frame_number,
                "label": str(row["label"]),
                "prob_fake": probability,
                "pred_fake": prediction,
                "correct": int(prediction == 1),
                "frame_path": str(image_path),
                "visualization_path": str(vis_path),
                "cam_path": str(raw_cam_path),
                "checkpoint_path": str(args.checkpoint),
                "model": args.model,
                "training_seed": args.seed,
                "sample_seed": args.sample_seed,
            })

            del image_bgr, image_rgb, cam

            if (len(records) % 20) == 0:
                logging.info("Generated CAMs: %d / %d", len(records), len(examples))

    finally:
        generator.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(records)
    results.to_csv(output_dir / "gradcam_examples.csv", index=False)

    counts = (
        results.groupby(["condition", "category"])
        .agg(n_frames=("sample_id", "count"),
             n_videos=("video_id", "nunique"))
        .reset_index()
    )
    counts.to_csv(output_dir / "sample_counts.csv", index=False)

    logging.info("Saved %d visualizations to %s", len(results), image_dir)
    logging.info("Metadata: %s", output_dir / "gradcam_examples.csv")
    logging.info("Sample counts:\n%s", counts.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()