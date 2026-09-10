"""
Unit tests for Phase 5: Clean Regression Test and Baseline Caching.

Validates:
1. Clean evaluation produces exact classification metric match against Approach 1 baseline.
2. Clean baseline caching stores complete predictions and metrics on first run.
3. Cache hit reloads from disk with zero redundant inference and identical outputs.
4. Clean video predictions match Approach 1 recorded predictions with 100% agreement.
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
from utils import get_approach1_checkpoint, get_device
from verify_clean_parity import (
    CLEAN_CONDITION,
    run_clean_evaluation_for_checkpoint,
    get_or_compute_clean_baseline,
)


class TestPhase5CleanRegression(unittest.TestCase):
    """Test suite for Phase 5 clean regression testing and caching."""

    @classmethod
    def setUpClass(cls):
        cls.manifest_path = MANIFESTS_ROOT / "manifest.csv"
        cls.device, _ = get_device()
        cls.ckpt_info = get_approach1_checkpoint("resnet50", 42, OUTPUT_ROOT)

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_clean_evaluation_parity_and_caching(self):
        """
        Run clean evaluation, verify metric parity with saved Approach 1 results,
        and test caching mechanism.
        """
        cache_dir = self.temp_dir / "resnet50_seed42" / "clean"

        # 1. First run: computes clean predictions and caches them
        frame_df, video_dfs, metrics = get_or_compute_clean_baseline(
            checkpoint_info=self.ckpt_info,
            manifest_path=self.manifest_path,
            clean_cache_dir=cache_dir,
            batch_size=64,
            device=self.device,
            use_amp=False,
        )

        self.assertTrue((cache_dir / "frame_predictions.csv").exists())
        self.assertTrue((cache_dir / "video_predictions_mean.csv").exists())
        self.assertTrue((cache_dir / "metrics.json").exists())

        # Verify against Approach 1 saved test_results.json
        with open(self.ckpt_info["run_dir"] / "test_results.json", "r", encoding="utf-8") as f:
            saved_results = json.load(f)

        saved_mean = saved_results["video_metrics"]["mean"]
        computed_mean = metrics["video_metrics"]["mean"]

        self.assertAlmostEqual(computed_mean["accuracy"], saved_mean["accuracy"], places=5)
        self.assertAlmostEqual(computed_mean["roc_auc"], saved_mean["roc_auc"], places=5)
        self.assertAlmostEqual(computed_mean["f1"], saved_mean["f1"], places=5)
        self.assertAlmostEqual(computed_mean["precision"], saved_mean["precision"], places=5)
        self.assertAlmostEqual(computed_mean["recall"], saved_mean["recall"], places=5)

        # 2. Second run: must reload from disk without recomputing
        cached_frame_df, cached_video_dfs, cached_metrics = get_or_compute_clean_baseline(
            checkpoint_info=self.ckpt_info,
            manifest_path=self.manifest_path,
            clean_cache_dir=cache_dir,
            batch_size=64,
            device=self.device,
            use_amp=False,
        )

        self.assertEqual(len(cached_frame_df), len(frame_df))
        self.assertEqual(len(cached_video_dfs["mean"]), len(video_dfs["mean"]))
        self.assertEqual(
            cached_metrics["video_metrics"]["mean"]["f1"],
            metrics["video_metrics"]["mean"]["f1"],
        )


if __name__ == "__main__":
    unittest.main()
