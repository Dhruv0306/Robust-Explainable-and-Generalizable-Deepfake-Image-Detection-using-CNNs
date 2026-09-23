"""
Validate completed Approach 3 outputs without rerunning Grad-CAM.

Usage:
    python scripts/validate_approach3_outputs.py \
        --root data/output/explainability_corrected_v1

The script is intentionally strict. It reports structural problems that should
be fixed before interpreting the experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


MODELS = {
    "xception": 2024,
    "efficientnet_b0": 2024,
    "resnet50": 123,
}

EXPECTED_FRAMES = 6304
EXPECTED_CONDITIONS = 13
EXPECTED_EVALS = EXPECTED_FRAMES * EXPECTED_CONDITIONS
EXPECTED_CATEGORY_VIDEO_UNITS = 60
EXPECTED_ORIGINAL_VIDEOS = 12
EXPECTED_MANIPULATED_SOURCE_VIDEOS = 12

KEY_FRAME = ["category", "video_id", "original_frame_number"]
KEY_EVAL = KEY_FRAME + ["transformation"]
KEY_VIDEO = [
    "model", "seed", "category", "video_id", "transformation"
]

EXPECTED_CONDITIONS_SET = {
    "clean",
    "jpeg_sev1", "jpeg_sev2", "jpeg_sev3",
    "resize_sev1", "resize_sev2", "resize_sev3",
    "darkening_sev1", "darkening_sev2", "darkening_sev3",
    "brightening_sev1", "brightening_sev2", "brightening_sev3",
}


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)
    print(f"[FAIL] {msg}")


def check_model(model: str, seed: int, root: Path, errors: list[str]) -> None:
    out = root / model / f"seed_{seed}"
    frame_path = out / "frame_level_results.csv"
    video_path = out / "video_level_results.csv"
    summary_path = out / "summary.json"

    print(f"\n=== {model} seed {seed} ===")

    for p in (frame_path, video_path, summary_path):
        if not p.exists():
            fail(f"Missing required artifact: {p}", errors)

    if not frame_path.exists() or not video_path.exists():
        return

    frame = pd.read_csv(frame_path, low_memory=False)
    video = pd.read_csv(video_path, low_memory=False)

    print(f"frame rows: {len(frame):,}")
    print(f"video rows: {len(video):,}")

    if len(frame) != EXPECTED_EVALS:
        fail(
            f"Expected {EXPECTED_EVALS:,} frame-condition rows, got {len(frame):,}",
            errors,
        )

    if frame[KEY_EVAL].duplicated().any():
        n = int(frame[KEY_EVAL].duplicated().sum())
        fail(f"{n} duplicate frame-condition keys", errors)

    if frame[KEY_FRAME].duplicated().any():
        # This is expected because each frame occurs once per condition.
        pass

    conditions = set(frame["transformation"].astype(str).unique())
    if conditions != EXPECTED_CONDITIONS_SET:
        fail(
            f"Condition mismatch. Missing={EXPECTED_CONDITIONS_SET - conditions}, "
            f"extra={conditions - EXPECTED_CONDITIONS_SET}",
            errors,
        )

    counts = frame.groupby("transformation").size()
    bad_counts = counts[counts != EXPECTED_FRAMES]
    if not bad_counts.empty:
        fail(f"Condition row-count mismatch:\n{bad_counts}", errors)

    category_video = frame[["category", "video_id"]].drop_duplicates()
    if len(category_video) != EXPECTED_CATEGORY_VIDEO_UNITS:
        fail(
            f"Expected {EXPECTED_CATEGORY_VIDEO_UNITS} category-video units, "
            f"got {len(category_video)}",
            errors,
        )

    # FaceForensics++ uses two ID formats in this test split:
    #   Original: 12 source video IDs such as ``033``
    #   Manipulated: 12 source-pair IDs such as ``033_097``
    # The fake-only source-video analysis later collapses the four
    # manipulation categories onto those 12 manipulated source-pair IDs.
    # Therefore, counting all video_id values together should yield 24, not 12.
    original_ids = set(
        frame.loc[frame["ground_truth"] == 0, "video_id"].astype(str).unique()
    )
    fake_ids = set(
        frame.loc[frame["ground_truth"] == 1, "video_id"].astype(str).unique()
    )
    all_ids = original_ids | fake_ids

    print(f"source video IDs: {len(all_ids)}")
    print(f"original source video IDs: {len(original_ids)}")
    print(f"manipulated source-pair IDs: {len(fake_ids)}")

    if len(original_ids) != EXPECTED_ORIGINAL_VIDEOS:
        fail(
            f"Expected {EXPECTED_ORIGINAL_VIDEOS} original source video IDs, "
            f"got {len(original_ids)}",
            errors,
        )

    if len(fake_ids) != EXPECTED_MANIPULATED_SOURCE_VIDEOS:
        fail(
            f"Expected {EXPECTED_MANIPULATED_SOURCE_VIDEOS} manipulated source-pair IDs, "
            f"got {len(fake_ids)}",
            errors,
        )

    if len(all_ids) != (EXPECTED_ORIGINAL_VIDEOS + EXPECTED_MANIPULATED_SOURCE_VIDEOS):
        fail(
            "Expected 24 total video IDs (12 original + 12 manipulated source-pair), "
            f"got {len(all_ids)}",
            errors,
        )

    # Every manipulated source pair should occur in all four manipulated
    # categories. This is the structure required by source-video aggregation.
    fake_category_video = frame.loc[
        frame["ground_truth"] == 1, ["category", "video_id"]
    ].drop_duplicates()
    categories_per_fake_id = fake_category_video.groupby("video_id")["category"].nunique()
    bad_pairs = categories_per_fake_id[
        categories_per_fake_id != 4
    ]
    if not bad_pairs.empty:
        fail(
            "Manipulated source-pair category coverage is incomplete:\n"
            f"{bad_pairs.to_string()}",
            errors,
        )

    if video[KEY_VIDEO].duplicated().any():
        n = int(video[KEY_VIDEO].duplicated().sum())
        fail(f"{n} duplicate video-level keys", errors)

    video_source = video[["video_id"]].drop_duplicates()
    print(f"video-level source IDs: {len(video_source)}")

    # Clean rows intentionally have no ES_cos or explanation_IoU because
    # stability is defined relative to a transformed condition.
    clean = video[video["transformation"] == "clean"]
    transformed = video[video["transformation"] != "clean"]

    if clean["ES_cos"].notna().any():
        fail("Clean video rows unexpectedly contain ES_cos", errors)

    if clean["explanation_IoU"].notna().any():
        fail("Clean video rows unexpectedly contain explanation_IoU", errors)

    for col in ["SO", "IoU", "saliency_mass", "faithfulness_blur"]:
        if col not in video.columns:
            fail(f"Missing expected metric column: {col}", errors)

    # A transformed condition should have stability values for valid fake rows.
    fake_trans = transformed[transformed["ground_truth"] == 1]
    for col in ["ES_cos", "explanation_IoU"]:
        valid = int(fake_trans[col].notna().sum())
        print(f"{col} transformed valid rows: {valid:,}")
        if valid == 0:
            fail(f"No valid transformed values for {col}", errors)

    # Failed frames should be represented by missing metrics, not missing rows.
    print(
        "transformed fake rows with missing SO:",
        int(fake_trans["SO"].isna().sum()),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []

    for model, seed in MODELS.items():
        check_model(model, seed, args.root, errors)

    print("\n=== Validation summary ===")
    if errors:
        print(f"{len(errors)} validation issue(s) found.")
        for e in errors:
            print(" -", e)
        return 1

    print("All structural validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
