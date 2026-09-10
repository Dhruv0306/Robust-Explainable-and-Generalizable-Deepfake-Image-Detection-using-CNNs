"""
Unit tests for Phase 6: Pilot Robustness Pipeline Verification.

Validates:
1. Pilot experiment runs on ResNet50 seed 42.
2. Configuration snapshots (config.json, config.txt) are created in experiment directory.
3. Output directory hierarchy matches Plan Section 49:
   <checkpoint>/<transformation>/sev{severity}/
4. Condition summary retains all required columns and performance deltas:
   delta_accuracy, delta_f1, delta_roc_auc.
5. Visual verification strip generator creates readable image panels.
"""
import unittest
import tempfile
import shutil
import json
from pathlib import Path
import pandas as pd
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import MANIFESTS_ROOT, OUTPUT_ROOT
from utils import get_approach1_checkpoint, get_device
from robustness_config import get_default_robustness_config
from run_robustness import run_robustness_for_checkpoint, run_robustness_experiment
from visual_samples import generate_pilot_visual_examples


class TestPhase6Pilot(unittest.TestCase):
    """Test suite for Phase 6 pilot robustness execution."""

    @classmethod
    def setUpClass(cls):
        cls.manifest_path = MANIFESTS_ROOT / "manifest.csv"
        cls.device, _ = get_device()
        cls.ckpt_info = get_approach1_checkpoint("resnet50", 42, OUTPUT_ROOT)

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_visual_verification_generation(self):
        """Verify visual example strips are generated and valid image files."""
        cfg = get_default_robustness_config()
        visual_dir = self.temp_dir / "figures" / "visual_examples"
        saved = generate_pilot_visual_examples(
            manifest_path=self.manifest_path,
            output_dir=visual_dir,
            config=cfg,
            max_samples=2,
        )
        self.assertGreaterEqual(len(saved), 8)
        for p in saved:
            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 5000)

    def test_pilot_directory_structure_and_delta_metrics(self):
        """
        Verify condition evaluation, folder layout, and delta metric calculations
        using a minimal 2-condition test configuration.
        """
        # Minimal test config: clean + 1 JPEG condition
        test_cfg = {
            "jpeg": {"enabled": True, "qualities": [80, 50, 20]},
            "resize": {"enabled": False, "scales": [0.75, 0.50, 0.25]},
            "brightness": {"enabled": False, "darker": [0.80, 0.60, 0.40], "brighter": [1.20, 1.40, 1.60]},
        }

        exp_dir = run_robustness_experiment(
            experiment_id="test_pilot_exp",
            output_root=self.temp_dir,
            manifest_path=self.manifest_path,
            checkpoints=[self.ckpt_info],
            config=test_cfg,
            batch_size=64,
            use_amp=False,
            save_visuals=False,
        )

        # 1. Check configuration snapshot
        self.assertTrue((exp_dir / "config.json").exists())
        self.assertTrue((exp_dir / "config.txt").exists())

        # 2. Check master summary
        master_csv = exp_dir / "master_summary.csv"
        self.assertTrue(master_csv.exists())
        master_df = pd.read_csv(master_csv)

        expected_cols = {
            "model", "seed", "condition_id", "transformation", "severity",
            "direction", "parameter_name", "parameter_value", "accuracy",
            "precision", "recall", "f1", "roc_auc", "delta_accuracy",
            "delta_f1", "delta_roc_auc",
        }
        self.assertTrue(expected_cols.issubset(set(master_df.columns)))

        # 3. Check condition directories
        ckpt_dir = exp_dir / "resnet50_seed42"
        self.assertTrue((ckpt_dir / "clean" / "metrics.json").exists())
        self.assertTrue((ckpt_dir / "jpeg" / "sev1" / "metrics.json").exists())
        self.assertTrue((ckpt_dir / "jpeg" / "sev2" / "metrics.json").exists())
        self.assertTrue((ckpt_dir / "jpeg" / "sev3" / "metrics.json").exists())

        # 4. Check clean baseline has 0 delta
        clean_row = master_df[master_df["condition_id"] == "clean"].iloc[0]
        self.assertEqual(clean_row["delta_f1"], 0.0)
        self.assertEqual(clean_row["delta_roc_auc"], 0.0)
        self.assertEqual(clean_row["delta_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
