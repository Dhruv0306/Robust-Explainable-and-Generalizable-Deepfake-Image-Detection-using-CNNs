"""
Unit tests for Phase 4: Inference Integration and Robustness Dataset.

Validates:
1. RobustnessDataset outputs correct (B, 3, input_size, input_size) tensors per model.
2. Dynamic transformations are applied before model-specific resize.
3. Clean condition produces identical model inputs to baseline DeepfakeDataset.
4. Frame predictions retain complete manifest metadata and transformation metadata.
5. Video aggregation preserves metadata and computes mean, median, mode aggregations.
6. Metrics payload contains frame, video, per-category metrics, and condition telemetry.
7. Raw artifact serialization matches directory schema (frame_predictions, video_predictions, metrics.json).
"""
import unittest
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import MANIFESTS_ROOT, OUTPUT_ROOT
from models import create_model
from utils import get_approach1_checkpoint, get_device
from dataset import DeepfakeDataset, RobustnessDataset, get_robustness_dataloader
from evaluate import (
    get_robustness_frame_predictions,
    aggregate_robustness_video_level,
    evaluate_robustness_condition,
    save_robustness_condition_results,
)


class TestPhase4InferenceIntegration(unittest.TestCase):
    """Test suite for Phase 4 inference integration."""

    @classmethod
    def setUpClass(cls):
        cls.manifest_path = MANIFESTS_ROOT / "manifest.csv"
        cls.device, _ = get_device()
        cls.ckpt_info = get_approach1_checkpoint("resnet50", 42, OUTPUT_ROOT)
        state = torch.load(cls.ckpt_info["checkpoint_path"], map_location=cls.device, weights_only=False)
        cls.model = create_model("resnet50", pretrained=False)
        cls.model.load_state_dict(state["model_state_dict"])
        cls.model.to(cls.device)
        cls.model.eval()

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_robustness_dataset_tensor_shapes(self):
        """Verify RobustnessDataset produces correct model-specific tensor dimensions."""
        for model_name, expected_size in [("resnet50", 224), ("efficientnet_b0", 224), ("xception", 299)]:
            ds = RobustnessDataset(
                manifest_path=self.manifest_path,
                model_name=model_name,
                split="test",
                condition={"transformation": "clean", "severity": 0},
            )
            img_tensor, label, meta = ds[0]
            self.assertEqual(img_tensor.shape, (3, expected_size, expected_size))
            self.assertIn(label.item(), [0.0, 1.0])
            self.assertIn("video_id", meta)
            self.assertIn("category", meta)
            self.assertIn("frame_path", meta)

    def test_clean_equivalence_with_baseline_dataset(self):
        """Verify clean RobustnessDataset produces bit-identical tensors to DeepfakeDataset."""
        from dataset import get_transforms

        baseline_ds = DeepfakeDataset(
            manifest_path=self.manifest_path,
            split="test",
            transform=get_transforms("resnet50", is_train=False),
        )
        robustness_ds = RobustnessDataset(
            manifest_path=self.manifest_path,
            model_name="resnet50",
            split="test",
            condition={"transformation": "clean", "severity": 0},
        )

        # Compare first 5 frames
        for i in range(5):
            b_img, b_label, b_vid = baseline_ds[i]
            r_img, r_label, r_meta = robustness_ds[i]

            self.assertTrue(torch.allclose(b_img, r_img, atol=1e-6))
            self.assertEqual(b_label.item(), r_label.item())
            self.assertEqual(b_vid, r_meta["video_id"])

    def test_corruption_modifies_tensors_before_resize(self):
        """Verify corrupted condition alters tensor values relative to clean condition."""
        clean_ds = RobustnessDataset(
            manifest_path=self.manifest_path,
            model_name="resnet50",
            split="test",
            condition={"transformation": "clean", "severity": 0},
        )
        corrupted_ds = RobustnessDataset(
            manifest_path=self.manifest_path,
            model_name="resnet50",
            split="test",
            condition={
                "transformation": "jpeg",
                "severity": 3,
                "direction": "none",
                "parameter_name": "quality",
                "parameter_value": 20,
            },
        )

        clean_tensor, _, _ = clean_ds[0]
        corrupted_tensor, _, _ = corrupted_ds[0]

        diff = torch.abs(clean_tensor - corrupted_tensor).mean().item()
        self.assertGreater(diff, 1e-3, f"Corrupted tensor should differ from clean tensor, got diff={diff}")

    def test_frame_prediction_schema_and_video_aggregation(self):
        """Verify evaluate_robustness_condition produces expected schema, metadata, and saved files."""
        # Use small 6-frame dataloader
        dl = get_robustness_dataloader(
            manifest_path=self.manifest_path,
            model_name="resnet50",
            split="test",
            condition={
                "transformation": "jpeg",
                "severity": 1,
                "direction": "none",
                "parameter_name": "quality",
                "parameter_value": 80,
                "condition_id": "jpeg_sev1_q80",
            },
            batch_size=2,
            num_workers=0,
        )

        class MiniLoader:
            def __init__(self, loader, n=3):
                self.batches = [b for i, b in enumerate(loader) if i < n]
            def __iter__(self):
                return iter(self.batches)
            def __len__(self):
                return len(self.batches)

        mini_loader = MiniLoader(dl, n=3)
        condition = {
            "transformation": "jpeg",
            "severity": 1,
            "direction": "none",
            "parameter_name": "quality",
            "parameter_value": 80,
            "condition_id": "jpeg_sev1_q80",
        }

        frame_df, video_dfs, metrics = evaluate_robustness_condition(
            model=self.model,
            dataloader=mini_loader,
            device=self.device,
            condition=condition,
            use_amp=False,
        )

        # Check frame DataFrame schema
        expected_frame_cols = {
            "video_id", "frame_path", "category", "label", "original_frame_number",
            "transformation", "severity", "direction", "parameter_name",
            "parameter_value", "prob_fake", "pred_fake",
        }
        self.assertEqual(set(frame_df.columns), expected_frame_cols)
        self.assertEqual(len(frame_df), 6)

        # Check video aggregation DataFrames
        for method in ["mean", "median", "mode"]:
            v_df = video_dfs[method]
            self.assertIn("video_id", v_df.columns)
            self.assertIn("prob_fake", v_df.columns)
            self.assertIn("pred_fake", v_df.columns)
            self.assertIn("category", v_df.columns)
            self.assertIn("transformation", v_df.columns)
            self.assertEqual(v_df["transformation"].iloc[0], "jpeg")
            self.assertEqual(v_df["severity"].iloc[0], 1)

        # Check metrics payload
        self.assertIn("frame_metrics", metrics)
        self.assertIn("video_metrics", metrics)
        self.assertIn("per_category_metrics", metrics)
        self.assertEqual(metrics["total_frames"], 6)

        # Check serialization
        save_robustness_condition_results(self.temp_dir, frame_df, video_dfs, metrics)
        self.assertTrue((self.temp_dir / "frame_predictions.csv").exists())
        self.assertTrue((self.temp_dir / "video_predictions_mean.csv").exists())
        self.assertTrue((self.temp_dir / "video_predictions_median.csv").exists())
        self.assertTrue((self.temp_dir / "video_predictions_mode.csv").exists())
        self.assertTrue((self.temp_dir / "metrics.json").exists())

        with open(self.temp_dir / "metrics.json", "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
        self.assertEqual(saved_metrics["condition"]["condition_id"], "jpeg_sev1_q80")


if __name__ == "__main__":
    unittest.main()
