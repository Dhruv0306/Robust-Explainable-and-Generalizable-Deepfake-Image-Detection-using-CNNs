from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HOLDOUTS = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}
MODELS = {"xception", "efficientnet_b0", "resnet50"}
SEEDS = {"42", "123", "2024"}
PREDICTION_FILENAME = "test_video_predictions_mean.csv"
REQUIRED_COLUMNS = {"video_id", "label", "category"}


def normalize_text(value: object) -> str:
    return str(value).strip().lower()


def normalize_label(value: object) -> str:
    """Map common real/fake label encodings to consistent names."""
    label = normalize_text(value)

    if label in {"0", "real", "original", "authentic"}:
        return "real"
    if label in {"1", "fake", "manipulated", "deepfake"}:
        return "fake"

    return label


def infer_run_key(csv_path: Path, root: Path) -> tuple[str, str, str] | None:
    """
    Infer (holdout, model, seed) from either folder layout:
      baseline: <holdout>/<model>_seed<seed>/...
      LOMO:     holdout_<holdout>_<model>_<...>_seed<seed>/...
    """
    try:
        parts = csv_path.relative_to(root).parts[:-1]
    except ValueError:
        return None

    path_text = "_".join(parts)
    path_lower = path_text.lower()

    # Baseline holdout is a standalone parent directory; LOMO uses
    # a folder name such as "holdout_Deepfakes_...".
    holdout = None
    for name in HOLDOUTS:
        name_lower = name.lower()
        if any(part.lower() == name_lower for part in parts):
            holdout = name
            break
        if re.search(
            rf"holdout[_-]{re.escape(name_lower)}(?:[_\\/]|$)",
            path_lower,
        ):
            holdout = name
            break

    # Match the model name as a complete token, allowing underscores
    # within names such as efficientnet_b0.
    model = next(
        (
            name
            for name in MODELS
            if re.search(
                rf"(?:^|[_\\/]){re.escape(name.lower())}(?=[_\\/]|$)",
                path_lower,
            )
        ),
        None,
    )

    seed_match = re.search(r"seed[_-]?(42|123|2024)(?:\D|$)", path_lower)
    seed = seed_match.group(1) if seed_match else None

    if holdout and model and seed:
        return holdout, model, seed

    return None


def find_prediction_files(root: Path) -> dict[tuple[str, str, str], Path]:
    files = sorted(root.rglob(PREDICTION_FILENAME))
    indexed: dict[tuple[str, str, str], Path] = {}

    for csv_path in files:
        key = infer_run_key(csv_path, root)

        if key is None:
            print(f"[WARN] Could not infer holdout/model/seed: {csv_path}")
            continue

        if key in indexed:
            raise ValueError(
                f"Duplicate prediction file for {key}:\n"
                f"  {indexed[key]}\n"
                f"  {csv_path}"
            )

        indexed[key] = csv_path

    return indexed


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["label_normalized"] = df["label"].map(normalize_label)
    df["category_normalized"] = df["category"].map(normalize_text)

    if df["video_id"].isna().any() or (df["video_id"] == "").any():
        raise ValueError(f"{path} contains blank video_id values.")

    duplicates = df.loc[df["video_id"].duplicated(), "video_id"].unique()
    if len(duplicates):
        raise ValueError(f"{path} has duplicate video IDs: {duplicates[:10].tolist()}")

    return df.sort_values("video_id").reset_index(drop=True)


def compare_pair(
    key: tuple[str, str, str],
    baseline_path: Path,
    lomo_path: Path,
) -> dict[str, object]:
    holdout, model, seed = key
    result: dict[str, object] = {
        "holdout": holdout,
        "model": model,
        "seed": seed,
        "baseline_file": str(baseline_path),
        "lomo_file": str(lomo_path),
        "baseline_videos": 0,
        "lomo_videos": 0,
        "missing_from_baseline": "",
        "missing_from_lomo": "",
        "label_mismatches": "",
        "category_mismatches": "",
        "status": "FAIL",
        "details": "",
    }

    try:
        baseline = read_predictions(baseline_path)
        lomo = read_predictions(lomo_path)

        baseline_ids = set(baseline["video_id"])
        lomo_ids = set(lomo["video_id"])

        result["baseline_videos"] = len(baseline_ids)
        result["lomo_videos"] = len(lomo_ids)
        result["missing_from_baseline"] = ";".join(sorted(lomo_ids - baseline_ids))
        result["missing_from_lomo"] = ";".join(sorted(baseline_ids - lomo_ids))

        common_ids = sorted(baseline_ids & lomo_ids)
        baseline_common = baseline.set_index("video_id").loc[common_ids]
        lomo_common = lomo.set_index("video_id").loc[common_ids]

        label_mismatches = [
            video_id
            for video_id in common_ids
            if baseline_common.at[video_id, "label_normalized"]
            != lomo_common.at[video_id, "label_normalized"]
        ]
        category_mismatches = [
            video_id
            for video_id in common_ids
            if baseline_common.at[video_id, "category_normalized"]
            != lomo_common.at[video_id, "category_normalized"]
        ]

        result["label_mismatches"] = ";".join(label_mismatches)
        result["category_mismatches"] = ";".join(category_mismatches)

        if (
            baseline_ids == lomo_ids
            and not label_mismatches
            and not category_mismatches
        ):
            result["status"] = "PASS"
        else:
            result["details"] = "Prediction IDs, labels, or categories differ."

    except Exception as exc:
        result["details"] = str(exc)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate alignment between baseline-on-LOMO-holdout and "
            "LOMO-trained mean video-level prediction CSVs."
        )
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("data/output/baseline_on_lomo_holdouts"),
    )
    parser.add_argument(
        "--lomo-root",
        type=Path,
        default=Path("data/output/lomo"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/output/baseline_on_lomo_holdouts/" "alignment_validation_report.csv"
        ),
    )
    args = parser.parse_args()

    if not args.baseline_root.is_dir():
        print(f"[ERROR] Baseline folder not found: {args.baseline_root}")
        return 2

    if not args.lomo_root.is_dir():
        print(f"[ERROR] LOMO folder not found: {args.lomo_root}")
        return 2

    baseline_files = find_prediction_files(args.baseline_root)
    lomo_files = find_prediction_files(args.lomo_root)

    expected_keys = {
        (holdout, model, seed)
        for holdout in HOLDOUTS
        for model in MODELS
        for seed in SEEDS
    }

    all_keys = sorted(expected_keys | set(baseline_files) | set(lomo_files))
    rows = []

    for key in all_keys:
        if key not in expected_keys:
            rows.append(
                {
                    "holdout": key[0],
                    "model": key[1],
                    "seed": key[2],
                    "baseline_file": str(baseline_files.get(key, "")),
                    "lomo_file": str(lomo_files.get(key, "")),
                    "baseline_videos": 0,
                    "lomo_videos": 0,
                    "missing_from_baseline": "",
                    "missing_from_lomo": "",
                    "label_mismatches": "",
                    "category_mismatches": "",
                    "status": "FAIL",
                    "details": "Unexpected holdout/model/seed combination.",
                }
            )
            continue

        baseline_path = baseline_files.get(key)
        lomo_path = lomo_files.get(key)

        if baseline_path is None or lomo_path is None:
            rows.append(
                {
                    "holdout": key[0],
                    "model": key[1],
                    "seed": key[2],
                    "baseline_file": str(baseline_path or ""),
                    "lomo_file": str(lomo_path or ""),
                    "baseline_videos": 0,
                    "lomo_videos": 0,
                    "missing_from_baseline": "",
                    "missing_from_lomo": "",
                    "label_mismatches": "",
                    "category_mismatches": "",
                    "status": "FAIL",
                    "details": (
                        "Missing baseline prediction file."
                        if baseline_path is None
                        else "Missing LOMO prediction file."
                    ),
                }
            )
            continue

        rows.append(compare_pair(key, baseline_path, lomo_path))

    report = pd.DataFrame(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)

    passed = int((report["status"] == "PASS").sum())
    failed = int((report["status"] != "PASS").sum())

    print(f"Expected combinations: {len(expected_keys)}")
    print(f"Baseline prediction files indexed: {len(baseline_files)}")
    print(f"LOMO prediction files indexed: {len(lomo_files)}")
    print(f"Alignment checks passed: {passed}")
    print(f"Alignment checks failed: {failed}")
    print(f"Report saved to: {args.report}")

    if failed:
        print("\nFailed checks:")
        print(
            report.loc[
                report["status"] != "PASS", ["holdout", "model", "seed", "details"]
            ].to_string(index=False)
        )
        return 1

    print("\nAll prediction pairs are aligned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
