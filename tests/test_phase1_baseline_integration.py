"""
Unit tests for Phase 1: Repository and Baseline Integration.
Validates:
1. All 9 Approach 1 checkpoints exist, load cleanly, and instantiate architectures.
2. Test manifest integrity, column schema, and split counts.
3. Model-specific preprocessing parameters (input sizes, ImageNet normalization).
4. Clean baseline parity against Approach 1 recorded evaluation metrics.
"""
import unittest
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import (
    OUTPUT_ROOT,
    MANIFESTS_ROOT,
    CATEGORIES,
    MODELS,
)
from utils import discover_approach1_checkpoints, get_approach1_checkpoint, get_device
from models import create_model, get_model_input_size, get_model_normalization
from dataset import get_transforms, get_dataloader
from evaluate import get_frame_predictions, aggregate_to_video_level, compute_metrics


class TestPhase1BaselineIntegration(unittest.TestCase):
    """Test suite for Phase 1 baseline integration."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = REPO_ROOT
        cls.manifest_path = MANIFESTS_ROOT / "manifest.csv"
        cls.checkpoints = discover_approach1_checkpoints(OUTPUT_ROOT)

    def test_checkpoint_discovery_count(self):
        """Verify all 9 checkpoints are discovered."""
        self.assertEqual(len(self.checkpoints), 9, f"Expected 9 checkpoints, found {len(self.checkpoints)}")

    def test_checkpoint_matrix_coverage(self):
        """Verify 3 architectures x 3 seeds {42, 123, 2024}."""
        expected_models = {"efficientnet_b0", "resnet50", "xception"}
        expected_seeds = {42, 123, 2024}

        discovered_pairs = set()
        for ckpt in self.checkpoints:
            discovered_pairs.add((ckpt["model"], ckpt["seed"]))

        for model in expected_models:
            for seed in expected_seeds:
                self.assertIn(
                    (model, seed),
                    discovered_pairs,
                    f"Missing checkpoint for model={model}, seed={seed}"
                )

    def test_checkpoints_load_state_dict(self):
        """Verify checkpoint files are valid binary weights and match architecture."""
        for ckpt in self.checkpoints:
            model_name = ckpt["model"]
            seed = ckpt["seed"]
            ckpt_path = ckpt["checkpoint_path"]

            self.assertTrue(ckpt_path.exists(), f"Checkpoint path does not exist: {ckpt_path}")
            # Ensure file is not an un-smudged LFS text pointer
            self.assertGreater(ckpt_path.stat().st_size, 10_000_000, f"File {ckpt_path} is suspiciously small (LFS pointer?)")

            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.assertIn("model_state_dict", state, f"model_state_dict missing in {ckpt_path}")

            model = create_model(model_name, pretrained=False)
            missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=True)
            self.assertEqual(len(missing), 0, f"Missing keys loading {model_name}: {missing}")
            self.assertEqual(len(unexpected), 0, f"Unexpected keys loading {model_name}: {unexpected}")

    def test_manifest_schema_and_integrity(self):
        """Verify test manifest exists, required columns present, and test frame count is 6304."""
        self.assertTrue(self.manifest_path.exists(), f"Manifest missing at {self.manifest_path}")
        df = pd.read_csv(self.manifest_path)

        required_cols = {"frame_path", "video_id", "category", "label", "original_frame_number", "split"}
        self.assertTrue(required_cols.issubset(set(df.columns)), f"Missing required columns in manifest: {required_cols - set(df.columns)}")

        test_df = df[df["split"] == "test"]
        self.assertEqual(len(test_df), 6304, f"Expected 6304 test frames, found {len(test_df)}")
        self.assertEqual(test_df["video_id"].nunique(), 24, f"Expected 24 test videos, found {test_df['video_id'].nunique()}")

        # Check all 5 manipulation categories are present in test set
        categories = set(test_df["category"].unique())
        expected_categories = {"Original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}
        self.assertEqual(categories, expected_categories, f"Category mismatch in test set: {categories} vs {expected_categories}")

        # Check binary labels in test set
        labels = set(test_df["label"].unique())
        self.assertEqual(labels, {"real", "fake"}, f"Unexpected labels in test set: {labels}")

    def test_model_specific_preprocessing_parameters(self):
        """Verify model-specific input sizes and ImageNet normalization."""
        self.assertEqual(get_model_input_size("resnet50"), 224)
        self.assertEqual(get_model_input_size("efficientnet_b0"), 224)
        self.assertEqual(get_model_input_size("xception"), 299)

        for model_name in ["resnet50", "efficientnet_b0", "xception"]:
            norm = get_model_normalization(model_name)
            self.assertEqual(norm["mean"], [0.485, 0.456, 0.406])
            self.assertEqual(norm["std"], [0.229, 0.224, 0.225])

            t = get_transforms(model_name, is_train=False)
            self.assertIsNotNone(t)

    def test_clean_baseline_parity(self):
        """
        Regression gate: evaluate ResNet50 Seed 42 on the test set
        and assert exact metric match with Approach 1 test_results.json.
        """
        device, _ = get_device()
        ckpt = get_approach1_checkpoint("resnet50", 42, OUTPUT_ROOT)

        state = torch.load(ckpt["checkpoint_path"], map_location=device, weights_only=False)
        model = create_model("resnet50", pretrained=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        model.eval()

        dl = get_dataloader(self.manifest_path, split="test", model_name="resnet50", batch_size=64, shuffle=False)
        preds_df = get_frame_predictions(model, dl, device=device, use_amp=False)

        video_mean = aggregate_to_video_level(preds_df, method="mean")
        metrics = compute_metrics(video_mean)

        results_json_path = ckpt["run_dir"] / "test_results.json"
        self.assertTrue(results_json_path.exists(), f"test_results.json not found at {results_json_path}")

        with open(results_json_path, "r") as f:
            saved_results = json.load(f)
        saved_mean = saved_results["video_metrics"]["mean"]

        self.assertAlmostEqual(metrics["accuracy"], saved_mean["accuracy"], places=5)
        self.assertAlmostEqual(metrics["roc_auc"], saved_mean["roc_auc"], places=5)
        self.assertAlmostEqual(metrics["f1"], saved_mean["f1"], places=5)


if __name__ == "__main__":
    unittest.main()
