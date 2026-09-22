"""
Approach 3 configuration.

The corrected run uses a fresh output namespace so results generated with the
previous cache identity are not reused.
"""

EXPLAINABILITY_OUTPUT_DIR = "explainability_corrected_v1"

EXPLAINABILITY_MODELS = ["xception", "efficientnet_b0", "resnet50"]

PRIMARY_SALIENCY_THRESHOLD = 0.20
SENSITIVITY_SALIENCY_THRESHOLDS = [0.10, 0.20, 0.30]

FAITHFULNESS_METHODS = ["blur", "zero", "mean"]
PRIMARY_FAITHFULNESS_METHOD = "blur"

# This is a descriptive threshold, not a significance threshold.
EXPLANATION_STABILITY_THRESHOLD = 0.80

MIN_SAMPLE_WILCOXON = 6
MIN_SAMPLE_SPEARMAN = 6
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42

FDR_FAMILIES = {
    "family_1_localization": ["SO", "IoU", "saliency_mass", "hit_rate"],
    "family_2_faithfulness": ["blur", "zero", "mean"],
    "family_3_stability": ["ES_cos", "explanation_IoU"],
    "family_4a_loc_faith": ["SO_vs_faith", "IoU_vs_faith", "saliency_mass_vs_faith"],
    "family_4b_det_exp_deg": [
        "Dprob_vs_DSO",
        "Dprob_vs_DIoU",
        "Dprob_vs_Dstability",
    ],
}

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


def get_mask_path_for_video(dataset_root, category, video_id):
    return (
        dataset_root
        / "manipulated_sequences"
        / category
        / "masks"
        / "videos"
        / f"{video_id}.mp4"
    )
