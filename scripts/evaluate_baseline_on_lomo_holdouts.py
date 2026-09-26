"""
Inference-only baseline evaluation on LOMO holdout test manifests.
Run from repository root:
    python scripts/evaluate_baseline_on_lomo_holdouts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "output"
LOMO_MANIFESTS = ROOT / "data" / "manifests" / "lomo"
DEST_ROOT = OUTPUT_ROOT / "baseline_on_lomo_holdouts"

MODELS = ("xception", "efficientnet_b0", "resnet50")
HOLDOUTS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
SEEDS = (42, 123, 2024)
BASELINE_BATCH = "2026-09-24"


def find_baseline_run(model: str, seed: int) -> Path:
    matches = []
    for folder in OUTPUT_ROOT.iterdir():
        if not folder.is_dir() or folder == OUTPUT_ROOT / "lomo":
            continue
        if not folder.name.lower().startswith(model + "_"):
            continue
        if BASELINE_BATCH not in folder.name:
            continue
        config_path = folder / "config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("seed") == seed:
            matches.append(folder)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one baseline run for {model}, seed={seed}; "
            f"found {[p.name for p in matches]}"
        )
    return matches[0]


def main() -> None:
    # The main guard is essential on Windows when DataLoader uses spawn workers.
    sys.path.insert(0, str(ROOT / "src"))
    from evaluate import evaluate_model

    jobs = []
    # Validate all inputs before beginning any inference.
    for holdout in HOLDOUTS:
        manifest = LOMO_MANIFESTS / f"manifest_holdout_{holdout}.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing holdout manifest: {manifest}")
        for model in MODELS:
            for seed in SEEDS:
                run_dir = find_baseline_run(model, seed)
                candidates = (
                    run_dir / "best_checkpoint.pth",
                    ROOT / "data" / "checkpoints" / run_dir.name / "best_checkpoint.pth",
                    ROOT / "data" / "checkpoints" / model / run_dir.name / "best_checkpoint.pth",
                )
                checkpoint = next((p for p in candidates if p.is_file()), None)
                if checkpoint is None:
                    raise FileNotFoundError(
                        f"Could not find best_checkpoint.pth for {run_dir.name}. "
                        "Add its actual location to the candidates tuple in this script."
                    )
                output_dir = DEST_ROOT / holdout / f"{model}_seed{seed}"
                jobs.append((holdout, model, seed, checkpoint, manifest, output_dir))

    print(f"Preflight passed: {len(jobs)} evaluations queued.")
    print(f"Results will be saved under: {DEST_ROOT}")
    for i, (holdout, model, seed, checkpoint, manifest, output_dir) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] model={model}, seed={seed}, holdout={holdout}")
        evaluate_model(
            model_name=model,
            checkpoint_path=checkpoint,
            manifest_path=manifest,
            output_dir=output_dir,
            split="test",
        )
    print("\nAll evaluations finished.")


if __name__ == "__main__":
    main()
