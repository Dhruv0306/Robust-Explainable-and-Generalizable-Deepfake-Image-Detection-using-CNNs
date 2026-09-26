import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path("data/output/lomo")
OUT = ROOT / "lomo_seed_metrics.csv"
SUMMARY = ROOT / "lomo_mean_sd_summary.csv"

HOLDOUTS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
MODELS = ["efficientnet_b0", "resnet50", "xception"]
SEEDS = [42, 123, 2024]

# Match the experiment identifier while allowing the device name and timestamp
# to vary between run folders.
PATTERN = re.compile(
    r"^holdout_(Deepfakes|Face2Face|FaceSwap|NeuralTextures)_"
    r"(efficientnet_b0|resnet50|xception)_.*_seed(42|123|2024)$"
)

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main():
    rows = []
    found = set()
    problems = []

    for folder in ROOT.iterdir():
        if not folder.is_dir():
            continue

        match = PATTERN.match(folder.name)
        if not match:
            continue

        holdout, model, seed_text = match.groups()
        seed = int(seed_text)
        key = (holdout, model, seed)

        if key in found:
            problems.append(f"Duplicate run for {key}: {folder.name}")
            continue
        found.add(key)

        result_path = folder / "test_results.json"
        checkpoint_path = folder / "best_checkpoint.pth"

        if not result_path.is_file():
            problems.append(f"Missing metrics file: {result_path}")
            continue
        if not checkpoint_path.is_file():
            problems.append(f"Missing checkpoint: {checkpoint_path}")

        try:
            with result_path.open("r", encoding="utf-8") as f:
                results = json.load(f)

            # The primary aggregation in the project plan is mean probability.
            metrics = results["video_metrics"]["mean"]
            row = {
                "holdout": holdout,
                "model": model,
                "seed": seed,
                "run_folder": folder.name,
            }
            for metric in METRICS:
                row[metric] = metrics.get(metric)
            rows.append(row)

        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            problems.append(f"Could not parse {result_path}: {exc}")

    expected = {
        (holdout, model, seed)
        for holdout in HOLDOUTS
        for model in MODELS
        for seed in SEEDS
    }
    missing = sorted(expected - found)

    if missing:
        problems.append(f"Missing run folders for: {missing}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["holdout", "model", "seed"])
        df.to_csv(OUT, index=False)

        summary_rows = []
        for (holdout, model), group in df.groupby(["holdout", "model"]):
            summary = {"holdout": holdout, "model": model, "n_seeds": len(group)}
            for metric in METRICS:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                summary[f"{metric}_mean"] = values.mean() if len(values) else None
                summary[f"{metric}_sd"] = (
                    values.std(ddof=1) if len(values) > 1 else None
                )
            summary_rows.append(summary)

        pd.DataFrame(summary_rows).sort_values(
            ["holdout", "model"]
        ).to_csv(SUMMARY, index=False)

    print(f"Matched run folders: {len(found)}")
    print(f"Parsed metrics files: {len(rows)}")
    print(f"Expected run combinations: {len(expected)}")
    print(f"Seed-level CSV: {OUT}")
    print(f"Mean/SD summary CSV: {SUMMARY}")

    if problems:
        print("\nAUDIT ISSUES:")
        for problem in problems:
            print(f"- {problem}")
    else:
        print("\nFolder and metrics-file audit passed.")
        print("Review the CSVs for metric completeness and plausible values.")


if __name__ == "__main__":
    main()