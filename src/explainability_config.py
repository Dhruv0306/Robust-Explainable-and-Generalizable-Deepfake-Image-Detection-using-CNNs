"""
Approach 3: Explainability and Explanation Robustness Configuration.
All explainability constants, thresholds, paths, and parameters.
"""
from pathlib import Path
from typing import Dict, Any, List

# Output root for explainability artifacts
EXPLAINABILITY_OUTPUT_DIR = "explainability"

# Approach 1 Model candidates
EXPLAINABILITY_MODELS = ["xception", "efficientnet_b0", "resnet50"]

# Saliency threshold percentiles (top 20% primary, 10% and 30% sensitivity)
PRIMARY_SALIENCY_THRESHOLD = 0.20
SENSITIVITY_SALIENCY_THRESHOLDS = [0.10, 0.20, 0.30]

# Faithfulness masking methods
FAITHFULNESS_METHODS = ["blur", "zero", "mean"]
PRIMARY_FAITHFULNESS_METHOD = "blur"

# Explanation stability threshold for binary descriptive grouping
EXPLANATION_STABILITY_THRESHOLD = 0.90

# Minimum sample size rules for statistical validity
MIN_SAMPLE_WILCOXON = 6   # min non-zero paired differences
MIN_SAMPLE_SPEARMAN = 6   # min paired observations
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42

# FDR correction families
FDR_FAMILIES = {
    "family_1_localization": ["SO", "IoU", "saliency_mass", "hit_rate"],
    "family_2_faithfulness": ["blur", "zero", "mean"],
    "family_3_stability": ["ES_cos", "explanation_IoU"],
    "family_4a_loc_faith": ["SO_vs_faith", "IoU_vs_faith", "saliency_mass_vs_faith"],
    "family_4b_det_exp_deg": [
        "DF1_vs_DSO",
        "DF1_vs_DIoU",
        "DF1_vs_Dstability",
        "DF1_vs_Dfaith",
    ],
}

# 12 Core transformations to evaluate (from Approach 2)
CORE_ROBUSTNESS_CONDITIONS = [
    ("clean", "clean", 0, None),
    ("jpeg", "jpeg", 1, 80),
    ("jpeg", "jpeg", 2, 50),
    ("jpeg", "jpeg", 3, 20),
    ("resize", "resize", 1, 0.75),
    ("resize", "resize", 2, 0.50),
    ("resize", "resize", 3, 0.25),
    ("brightness_dark", "darkening", 1, 0.80),
    ("brightness_dark", "darkening", 2, 0.60),
    ("brightness_dark", "darkening", 3, 0.40),
    ("brightness_bright", "brightening", 1, 1.20),
    ("brightness_bright", "brightening", 2, 1.40),
    ("brightness_bright", "brightening", 3, 1.60),
]


def get_mask_path_for_video(dataset_root: Path, category: str, video_id: str) -> Path:
    """
    Resolve path to FaceForensics++ ground-truth manipulation mask video.

    Args:
        dataset_root: Path to data/datasets/FaceForensics++
        category: 'Deepfakes', 'Face2Face', 'FaceSwap', or 'NeuralTextures'
        video_id: e.g. '033_097'

    Returns:
        Path to mask video (e.g., .../Deepfakes/masks/videos/033_097.mp4)
    """
    return dataset_root / "manipulated_sequences" / category / "masks" / "videos" / f"{video_id}.mp4"
