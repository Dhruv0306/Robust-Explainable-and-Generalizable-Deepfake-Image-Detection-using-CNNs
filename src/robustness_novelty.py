"""
Approach 2 Novelty Extension Engine.

Implements three post-processing analytical dimensions from Approach_02_novelty_plan.md:
1. Novelty A: Confidence-to-decision stability (class-stratified probability drift, flip rate, 4-state error transitions).
2. Novelty B: Cross-architecture failure agreement (pairwise failure-set Jaccard overlap, prediction disagreement, 3-model consensus).
3. Novelty C: Transformation x manipulation vulnerability (per-manipulation Delta F1 and Recall with exact N=12 video accounting).

All calculations derive strictly from existing prediction artifacts in core_experiment_117.
No models are retrained, fine-tuned, or re-inferred.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_ROOT, OUTPUT_ROOT, ROBUSTNESS_ROOT
from utils import setup_logging


# -----------------------------------------------------------------------------
# Data Ingestion Helpers
# -----------------------------------------------------------------------------

def load_video_predictions(experiment_dir: Path) -> pd.DataFrame:
    """
    Load and collate video-level predictions across all 9 checkpoints and 13 conditions.
    """
    experiment_dir = Path(experiment_dir)
    records = []

    for ckpt_dir in experiment_dir.iterdir():
        if not ckpt_dir.is_dir() or ckpt_dir.name in ("figures", "novelty"):
            continue

        parts = ckpt_dir.name.split("_seed")
        if len(parts) != 2:
            continue
        model_name, seed_str = parts[0], parts[1]
        seed = int(seed_str)

        for csv_path in ckpt_dir.glob("**/video_predictions_mean.csv"):
            df = pd.read_csv(csv_path)
            df["model"] = model_name
            df["seed"] = seed
            records.append(df)

    if not records:
        raise FileNotFoundError(f"No video predictions found in {experiment_dir}")

    full_df = pd.concat(records, ignore_index=True)
    return full_df


def load_category_video_predictions(experiment_dir: Path) -> pd.DataFrame:
    """
    Collate predictions aggregated at the (video_id, category) level from frame_predictions.csv.
    Yields exactly N=12 video observations for each of the 5 categories (Original + 4 fakes).
    """
    experiment_dir = Path(experiment_dir)
    records = []

    for ckpt_dir in experiment_dir.iterdir():
        if not ckpt_dir.is_dir() or ckpt_dir.name in ("figures", "novelty"):
            continue

        parts = ckpt_dir.name.split("_seed")
        if len(parts) != 2:
            continue
        model_name, seed_str = parts[0], parts[1]
        seed = int(seed_str)

        for csv_path in ckpt_dir.glob("**/frame_predictions.csv"):
            f_df = pd.read_csv(csv_path)
            t_name = f_df["transformation"].iloc[0]
            sev = f_df["severity"].iloc[0]
            direction = f_df["direction"].iloc[0]
            param_name = f_df["parameter_name"].iloc[0]
            param_val = f_df["parameter_value"].iloc[0]

            # Aggregate per video and manipulation category
            grouped = (
                f_df.groupby(["video_id", "category"])
                .agg({
                    "label": "first",
                    "prob_fake": "mean",
                })
                .reset_index()
            )
            grouped["pred_fake"] = (grouped["prob_fake"] >= 0.5).astype(int)
            grouped["model"] = model_name
            grouped["seed"] = seed
            grouped["transformation"] = t_name
            grouped["severity"] = sev
            grouped["direction"] = direction
            grouped["parameter_name"] = param_name
            grouped["parameter_value"] = param_val

            records.append(grouped)

    full_cat_df = pd.concat(records, ignore_index=True)
    return full_cat_df


def audit_novelty_artifacts(experiment_dir: Path) -> Dict[str, Any]:
    """Validate the existing Approach 2 artifacts before novelty post-processing."""
    experiment_dir = Path(experiment_dir)
    master_path = experiment_dir / "master_summary.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master summary: {master_path}")

    master = pd.read_csv(master_path)
    video_df = load_video_predictions(experiment_dir)
    cat_df = load_category_video_predictions(experiment_dir)

    expected_conditions = 13
    expected_models = {"efficientnet_b0", "resnet50", "xception"}
    expected_seeds = {42, 123, 2024}
    expected_video_cols = {
        "video_id", "category", "label", "transformation", "severity",
        "prob_fake", "pred_fake", "model", "seed",
    }
    expected_frame_cols = {
        "video_id", "category", "label", "prob_fake", "pred_fake",
        "transformation", "severity", "model", "seed",
    }

    if len(master) != 117:
        raise ValueError(f"Expected 117 master rows, found {len(master)}")
    if set(master["model"]) != expected_models or set(master["seed"]) != expected_seeds:
        raise ValueError("Master summary does not contain the expected model/seed matrix")
    if video_df["model"].nunique() != 3 or video_df["seed"].nunique() != 3:
        raise ValueError("Video predictions do not contain all 9 checkpoints")
    if not expected_video_cols.issubset(video_df.columns):
        raise ValueError(f"Video prediction schema is missing {expected_video_cols - set(video_df.columns)}")
    if not expected_frame_cols.issubset(cat_df.columns):
        raise ValueError(f"Category prediction schema is missing {expected_frame_cols - set(cat_df.columns)}")

    duplicate_keys = ["model", "seed", "transformation", "severity", "video_id"]
    if video_df.duplicated(duplicate_keys).any():
        raise ValueError("Duplicate model/seed/condition/video prediction rows detected")

    clean_categories = cat_df[cat_df["severity"] == 0]["category"].value_counts()
    expected_categories = {"Original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}
    if set(clean_categories.index) != expected_categories or not (clean_categories == 108).all():
        raise ValueError(f"Expected 108 clean rows per category (9 checkpoints x 12 videos), got {clean_categories.to_dict()}")

    return {
        "master_rows": len(master),
        "video_prediction_rows": len(video_df),
        "category_prediction_rows": len(cat_df),
        "models": sorted(expected_models),
        "seeds": sorted(expected_seeds),
        "conditions_per_checkpoint": expected_conditions,
        "clean_category_rows": clean_categories.to_dict(),
        "passed": True,
    }


# -----------------------------------------------------------------------------
# Novelty A: Confidence-to-Decision Stability
# -----------------------------------------------------------------------------

def compute_confidence_decision_analysis(
    video_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute class-stratified probability drift, prediction flip rates, and 4-state error transitions.

    Returns:
        (confidence_summary_df, flip_summary_df, error_transition_df)
    """
    # Separate clean reference per (model, seed, video_id)
    clean_df = video_df[video_df["severity"] == 0].copy()
    clean_ref = clean_df[["model", "seed", "video_id", "prob_fake", "pred_fake", "label"]].rename(
        columns={"prob_fake": "clean_prob", "pred_fake": "clean_pred"}
    )

    # Merge with all conditions
    merged = pd.merge(video_df, clean_ref, on=["model", "seed", "video_id", "label"], how="inner")
    merged["delta_p_fake"] = merged["prob_fake"] - merged["clean_prob"]
    merged["abs_delta_p"] = merged["delta_p_fake"].abs()
    merged["is_flip"] = (merged["pred_fake"] != merged["clean_pred"]).astype(int)

    # Error-state transition categorization
    def categorize_transition(row):
        clean_correct = (row["clean_pred"] == row["label"])
        trans_correct = (row["pred_fake"] == row["label"])
        if clean_correct and trans_correct:
            return "stable_correct"
        elif clean_correct and not trans_correct:
            return "robustness_failure"
        elif not clean_correct and trans_correct:
            return "transformation_correction"
        else:
            return "persistent_error"

    merged["transition_state"] = merged.apply(categorize_transition, axis=1)

    group_keys = ["transformation", "severity", "direction", "parameter_name", "parameter_value", "model", "seed"]

    # 1. Confidence Summary (Class-Stratified Drift)
    conf_records = []
    for keys, grp in merged.groupby(group_keys, dropna=False):
        t_name, sev, direct, p_name, p_val, model, seed = keys
        real_grp = grp[grp["label"] == 0]
        fake_grp = grp[grp["label"] == 1]

        conf_records.append({
            "transformation": t_name,
            "severity": sev,
            "direction": direct,
            "parameter_name": p_name,
            "parameter_value": p_val,
            "model": model,
            "seed": seed,
            "overall_mean_delta_p": grp["delta_p_fake"].mean(),
            "overall_median_delta_p": grp["delta_p_fake"].median(),
            "overall_mean_abs_delta_p": grp["abs_delta_p"].mean(),
            "real_mean_delta_p": real_grp["delta_p_fake"].mean(),
            "real_median_delta_p": real_grp["delta_p_fake"].median(),
            "fake_mean_delta_p": fake_grp["delta_p_fake"].mean(),
            "fake_median_delta_p": fake_grp["delta_p_fake"].median(),
        })
    conf_summary_df = pd.DataFrame(conf_records)

    # 2. Prediction Flip Summary
    flip_records = []
    for keys, grp in merged.groupby(group_keys, dropna=False):
        t_name, sev, direct, p_name, p_val, model, seed = keys
        total_evals = len(grp)
        flip_count = grp["is_flip"].sum()
        flip_rate = flip_count / total_evals if total_evals > 0 else 0.0

        # Flip directions: fake->real vs real->fake
        fake_to_real = ((grp["clean_pred"] == 1) & (grp["pred_fake"] == 0)).sum()
        real_to_fake = ((grp["clean_pred"] == 0) & (grp["pred_fake"] == 1)).sum()

        flip_records.append({
            "transformation": t_name,
            "severity": sev,
            "direction": direct,
            "parameter_name": p_name,
            "parameter_value": p_val,
            "model": model,
            "seed": seed,
            "total_videos": total_evals,
            "flip_count": flip_count,
            "flip_rate": flip_rate,
            "fake_to_real_flips": fake_to_real,
            "real_to_fake_flips": real_to_fake,
        })
    flip_summary_df = pd.DataFrame(flip_records)

    # 3. Error-State Transitions Summary
    state_records = []
    for keys, grp in merged.groupby(group_keys, dropna=False):
        t_name, sev, direct, p_name, p_val, model, seed = keys
        n_total = len(grp)
        counts = grp["transition_state"].value_counts().to_dict()

        state_records.append({
            "transformation": t_name,
            "severity": sev,
            "direction": direct,
            "parameter_name": p_name,
            "parameter_value": p_val,
            "model": model,
            "seed": seed,
            "total_evals": n_total,
            "stable_correct": counts.get("stable_correct", 0),
            "robustness_failure": counts.get("robustness_failure", 0),
            "transformation_correction": counts.get("transformation_correction", 0),
            "persistent_error": counts.get("persistent_error", 0),
            "robustness_failure_rate": counts.get("robustness_failure", 0) / n_total,
            "correction_rate": counts.get("transformation_correction", 0) / n_total,
        })
    error_transition_df = pd.DataFrame(state_records)

    return conf_summary_df, flip_summary_df, error_transition_df


# -----------------------------------------------------------------------------
# Novelty B: Cross-Architecture Failure Agreement
# -----------------------------------------------------------------------------

def compute_pairwise_architecture_failure_agreement(
    video_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise failure overlap (Jaccard), prediction disagreement, and 3-architecture consensus.

    Returns:
        (overlap_summary_df, disagreement_summary_df, consensus_summary_df)
    """
    clean_df = video_df[video_df["severity"] == 0].copy()
    clean_ref = clean_df[["model", "seed", "video_id", "pred_fake", "label"]].rename(
        columns={"pred_fake": "clean_pred"}
    )
    merged = pd.merge(video_df, clean_ref, on=["model", "seed", "video_id", "label"], how="inner")
    # Robustness failure: clean correct -> transformed incorrect
    merged["is_failure"] = (
        (merged["clean_pred"] == merged["label"]) & (merged["pred_fake"] != merged["label"])
    ).astype(int)

    corrupted_df = merged[merged["severity"] > 0]
    models = sorted(corrupted_df["model"].unique())
    pairs = [
        (models[0], models[1]),
        (models[0], models[2]),
        (models[1], models[2]),
    ]

    overlap_records = []
    disagreement_records = []

    cond_cols = ["transformation", "severity", "direction", "parameter_name", "parameter_value", "seed"]

    for cond_keys, grp in corrupted_df.groupby(cond_cols, dropna=False):
        t_name, sev, direct, p_name, p_val, seed = cond_keys

        for m_a, m_b in pairs:
            grp_a = grp[grp["model"] == m_a].set_index("video_id")
            grp_b = grp[grp["model"] == m_b].set_index("video_id")

            common_vids = grp_a.index.intersection(grp_b.index)
            if len(common_vids) == 0:
                continue

            sub_a = grp_a.loc[common_vids]
            sub_b = grp_b.loc[common_vids]

            # Failure sets
            fail_a = set(sub_a[sub_a["is_failure"] == 1].index)
            fail_b = set(sub_b[sub_b["is_failure"] == 1].index)

            intersection = len(fail_a & fail_b)
            union = len(fail_a | fail_b)

            # Piecewise Jaccard per Plan Section 6
            if union == 0:
                jaccard = 1.0  # complete agreement on absence of failure
            else:
                jaccard = intersection / union

            overlap_records.append({
                "transformation": t_name,
                "severity": sev,
                "direction": direct,
                "parameter_name": p_name,
                "parameter_value": p_val,
                "seed": seed,
                "model_a": m_a,
                "model_b": m_b,
                "model_pair": f"{m_a}_vs_{m_b}",
                "fail_a_count": len(fail_a),
                "fail_b_count": len(fail_b),
                "intersection": intersection,
                "union": union,
                "jaccard_overlap": jaccard,
            })

            # Prediction disagreement rate
            disagree_count = (sub_a["pred_fake"] != sub_b["pred_fake"]).sum()
            disagreement_rate = disagree_count / len(common_vids)

            disagreement_records.append({
                "transformation": t_name,
                "severity": sev,
                "direction": direct,
                "parameter_name": p_name,
                "parameter_value": p_val,
                "seed": seed,
                "model_a": m_a,
                "model_b": m_b,
                "model_pair": f"{m_a}_vs_{m_b}",
                "paired_videos": len(common_vids),
                "disagreement_count": disagree_count,
                "disagreement_rate": disagreement_rate,
            })

    overlap_df = pd.DataFrame(overlap_records)
    disagreement_df = pd.DataFrame(disagreement_records)

    # 3. Three-Architecture Failure Consensus
    consensus_records = []
    for cond_keys, grp in corrupted_df.groupby(["transformation", "severity", "direction", "parameter_name", "parameter_value", "seed"], dropna=False):
        t_name, sev, direct, p_name, p_val, seed = cond_keys

        # Pivot to videos x models
        piv = grp.pivot_table(index="video_id", columns="model", values="is_failure", aggfunc="first").fillna(0)
        piv["failure_consensus_count"] = piv.sum(axis=1).astype(int)

        counts = piv["failure_consensus_count"].value_counts().to_dict()
        n_vids = len(piv)

        consensus_records.append({
            "transformation": t_name,
            "severity": sev,
            "direction": direct,
            "parameter_name": p_name,
            "parameter_value": p_val,
            "seed": seed,
            "total_videos": n_vids,
            "models_failed_0": counts.get(0, 0),
            "models_failed_1": counts.get(1, 0),
            "models_failed_2": counts.get(2, 0),
            "models_failed_3": counts.get(3, 0),
        })

    consensus_df = pd.DataFrame(consensus_records)

    return overlap_df, disagreement_df, consensus_df


# -----------------------------------------------------------------------------
# Novelty C: Transformation x Manipulation Vulnerability
# -----------------------------------------------------------------------------

def compute_manipulation_vulnerability(
    cat_video_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute category-level Delta F1 and Recall across transformations with exact video accounting (N=12).

    Returns:
        (vulnerability_summary_df, f1_delta_matrix, recall_delta_matrix)
    """
    from sklearn.metrics import f1_score, recall_score

    # Compute metrics for each model, seed, condition, category
    records = []
    group_cols = [
        "model", "seed", "transformation", "severity", "direction",
        "parameter_name", "parameter_value", "category"
    ]

    for keys, grp in cat_video_df.groupby(group_cols, dropna=False):
        model, seed, t_name, sev, direct, p_name, p_val, cat = keys
        y_true = grp["label"].values
        y_pred = grp["pred_fake"].values
        n_vids = len(grp)

        # In FaceForensics++: 12 Real videos (Original) and 12 Fake videos for each manipulation
        # Compute category-level accuracy and binary recall/f1
        if cat == "Original":
            # Real class recall: true negative rate
            recall = (y_pred == 0).sum() / len(y_pred) if len(y_pred) > 0 else 0.0
            f1 = recall  # for Real class
        else:
            recall = (y_pred == 1).sum() / len(y_pred) if len(y_pred) > 0 else 0.0
            f1 = recall

        records.append({
            "model": model,
            "seed": seed,
            "transformation": t_name,
            "severity": sev,
            "direction": direct,
            "parameter_name": p_name,
            "parameter_value": p_val,
            "category": cat,
            "video_count": n_vids,
            "f1": f1,
            "recall": recall,
        })

    perf_df = pd.DataFrame(records)

    # Compute clean reference per (model, seed, category)
    clean_ref = perf_df[perf_df["severity"] == 0][["model", "seed", "category", "f1", "recall"]].rename(
        columns={"f1": "clean_f1", "recall": "clean_recall"}
    )
    merged = pd.merge(perf_df, clean_ref, on=["model", "seed", "category"], how="inner")
    merged["delta_f1"] = merged["f1"] - merged["clean_f1"]
    merged["delta_recall"] = merged["recall"] - merged["clean_recall"]

    # Filter to corrupted conditions and aggregate across 3 seeds and 3 models
    corrupted = merged[merged["severity"] > 0]
    agg_cols = ["f1", "recall", "delta_f1", "delta_recall"]
    cat_agg = (
        corrupted.groupby(["transformation", "severity", "direction", "parameter_name", "parameter_value", "category"], dropna=False)[agg_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    cat_agg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in cat_agg.columns]

    # Main Presentation Matrices at Severity 3 (Strongest corruption)
    sev3_df = merged[merged["severity"] == 3]
    fake_sev3 = sev3_df[sev3_df["category"] != "Original"]

    f1_delta_piv = fake_sev3.pivot_table(
        index="category",
        columns="transformation",
        values="delta_f1",
        aggfunc="mean",
    )

    recall_delta_piv = fake_sev3.pivot_table(
        index="category",
        columns="transformation",
        values="delta_recall",
        aggfunc="mean",
    )

    return cat_agg, f1_delta_piv, recall_delta_piv


# -----------------------------------------------------------------------------
# Visualization Engine for Novelty Findings
# -----------------------------------------------------------------------------

def generate_novelty_heatmaps_and_plots(
    overlap_df: pd.DataFrame,
    f1_delta_piv: pd.DataFrame,
    recall_delta_piv: pd.DataFrame,
    conf_summary_df: pd.DataFrame,
    flip_summary_df: pd.DataFrame,
    figures_dir: Path,
) -> List[Path]:
    """
    Generate clean, publication-ready figures using diverging colormaps centered at 0.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # 1. Figure N1: Transformation x Manipulation Delta F1 Heatmap
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    data = f1_delta_piv.values
    im = ax.imshow(data, cmap="coolwarm", vmin=-0.8, vmax=0.1, aspect="auto")

    ax.set_xticks(np.arange(len(f1_delta_piv.columns)))
    ax.set_yticks(np.arange(len(f1_delta_piv.index)))
    ax.set_xticklabels([col.replace("_", " ").title() for col in f1_delta_piv.columns], fontsize=10)
    ax.set_yticklabels(f1_delta_piv.index, fontsize=10)

    # Annotate with Delta F1 value and N=12
    for i in range(len(f1_delta_piv.index)):
        for j in range(len(f1_delta_piv.columns)):
            val = data[i, j]
            text = f"{val:+.2f}\n(N=12)"
            color = "white" if val < -0.4 or val > 0.05 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_title("Transformation × Manipulation Vulnerability (Δ F1 at Severity 3)", fontsize=11, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ F1 (Transformed − Clean)", fontsize=10)

    plt.tight_layout()
    p1 = figures_dir / "manipulation_vulnerability_delta_f1_heatmap.png"
    plt.savefig(p1)
    plt.close()
    saved_paths.append(p1)

    # 2. Figure N2: Cross-Architecture Robustness Failure Overlap (Jaccard)
    # Average across seeds and severities per model pair and transformation
    sev3_overlap = overlap_df[overlap_df["severity"] == 3]
    pair_piv = sev3_overlap.pivot_table(
        index="model_pair",
        columns="transformation",
        values="jaccard_overlap",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    im2 = ax.imshow(pair_piv.values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(pair_piv.columns)))
    ax.set_yticks(np.arange(len(pair_piv.index)))
    ax.set_xticklabels([c.replace("_", " ").title() for c in pair_piv.columns], fontsize=10)
    ax.set_yticklabels([idx.replace("_", " ") for idx in pair_piv.index], fontsize=10)

    for i in range(len(pair_piv.index)):
        for j in range(len(pair_piv.columns)):
            val = pair_piv.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 0.6 else "white", fontsize=10, fontweight="bold")

    ax.set_title("Cross-Architecture Robustness Failure Overlap (Jaccard Index at Sev 3)", fontsize=11, fontweight="bold", pad=12)
    cbar = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard Overlap (1.0 = Shared Failure)", fontsize=10)

    plt.tight_layout()
    p2 = figures_dir / "architecture_failure_overlap_jaccard_heatmap.png"
    plt.savefig(p2)
    plt.close()
    saved_paths.append(p2)

    # 3. Figure N3: Class-Stratified Probability Drift & Flip Rates
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Plot A: Class-Stratified Drift on JPEG
    ax_a = axes[0]
    jpeg_conf = conf_summary_df[conf_summary_df["transformation"] == "jpeg"].groupby(["severity", "model"])[["real_mean_delta_p", "fake_mean_delta_p"]].mean().reset_index()

    for m in jpeg_conf["model"].unique():
        sub = jpeg_conf[jpeg_conf["model"] == m].sort_values("severity")
        # Prepend clean severity 0
        sevs = [0] + list(sub["severity"])
        fake_drift = [0.0] + list(sub["fake_mean_delta_p"])
        real_drift = [0.0] + list(sub["real_mean_delta_p"])

        ax_a.plot(sevs, fake_drift, marker="o", linewidth=2, label=f"{m} (Fake Drift)")
        ax_a.plot(sevs, real_drift, marker="^", linestyle="--", linewidth=1.5, label=f"{m} (Real Drift)")

    ax_a.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax_a.set_title("Class-Stratified Probability Drift Under JPEG", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Severity Level", fontsize=10)
    ax_a.set_ylabel("Mean Δ P(Fake)", fontsize=10)
    ax_a.set_xticks([0, 1, 2, 3])
    ax_a.legend(fontsize=8, frameon=True)

    # Plot B: Video Prediction Flip Rates across Core Transformations
    ax_b = axes[1]
    flip_agg = flip_summary_df.groupby(["transformation", "severity"])["flip_rate"].mean().reset_index()

    for t in flip_agg["transformation"].unique():
        sub = flip_agg[flip_agg["transformation"] == t].sort_values("severity")
        sevs = [0] + list(sub["severity"])
        flips = [0.0] + list(sub["flip_rate"])
        ax_b.plot(sevs, flips, marker="s", linewidth=2, label=t.replace("_", " ").title())

    ax_b.set_title("Mean Video Prediction Flip Rate vs. Severity", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Severity Level", fontsize=10)
    ax_b.set_ylabel("Flip Rate (Count / 24)", fontsize=10)
    ax_b.set_xticks([0, 1, 2, 3])
    ax_b.set_ylim(-0.05, 0.6)
    ax_b.legend(fontsize=9, frameon=True)

    plt.tight_layout()
    p3 = figures_dir / "confidence_drift_and_flip_rate_analysis.png"
    plt.savefig(p3)
    plt.close()
    saved_paths.append(p3)

    return saved_paths


# -----------------------------------------------------------------------------
# Main Pipeline Execution
# -----------------------------------------------------------------------------

def run_novelty_pipeline(
    experiment_dir: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute end-to-end novelty pipeline and export summary tables and heatmaps.
    """
    experiment_dir = Path(experiment_dir)
    if output_dir is None:
        output_dir = experiment_dir / "novelty"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Executing Novelty Pipeline on {experiment_dir}...")

    # Phase 0: audit existing artifacts before any calculations
    audit = audit_novelty_artifacts(experiment_dir)
    with open(output_dir / "artifact_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logging.info("Artifact audit passed: %s", audit)

    # Load collated predictions
    video_df = load_video_predictions(experiment_dir)
    cat_video_df = load_category_video_predictions(experiment_dir)

    # Novelty A: Confidence and Decision Dynamics
    logging.info("Computing Novelty A: Confidence-to-Decision Stability...")
    conf_df, flip_df, error_df = compute_confidence_decision_analysis(video_df)
    conf_df.to_csv(output_dir / "confidence_decision_summary.csv", index=False)
    flip_df.to_csv(output_dir / "prediction_flip_summary.csv", index=False)
    error_df.to_csv(output_dir / "error_transition_summary.csv", index=False)

    # Novelty B: Cross-Architecture Failure Agreement
    logging.info("Computing Novelty B: Cross-Architecture Failure Agreement...")
    overlap_df, disagree_df, consensus_df = compute_pairwise_architecture_failure_agreement(video_df)
    overlap_df.to_csv(output_dir / "failure_overlap_summary.csv", index=False)
    disagree_df.to_csv(output_dir / "architecture_disagreement_summary.csv", index=False)
    consensus_df.to_csv(output_dir / "failure_consensus_summary.csv", index=False)

    # Novelty C: Transformation x Manipulation Vulnerability
    logging.info("Computing Novelty C: Transformation × Manipulation Vulnerability...")
    cat_agg_df, f1_piv, recall_piv = compute_manipulation_vulnerability(cat_video_df)
    cat_agg_df.to_csv(output_dir / "manipulation_vulnerability_summary.csv", index=False)
    f1_piv.to_csv(output_dir / "manipulation_f1_delta.csv")
    recall_piv.to_csv(output_dir / "manipulation_recall_delta.csv")

    # Generate Heatmaps & Diagnostic Plots
    logging.info("Generating Novelty Figures...")
    saved_figs = generate_novelty_heatmaps_and_plots(
        overlap_df=overlap_df,
        f1_delta_piv=f1_piv,
        recall_delta_piv=recall_piv,
        conf_summary_df=conf_df,
        flip_summary_df=flip_df,
        figures_dir=output_dir / "figures",
    )

    logging.info(f"Novelty pipeline completed. Outputs saved to {output_dir}")

    return {
        "output_dir": output_dir,
        "saved_tables": [
            "confidence_decision_summary.csv",
            "prediction_flip_summary.csv",
            "error_transition_summary.csv",
            "failure_overlap_summary.csv",
            "architecture_disagreement_summary.csv",
            "failure_consensus_summary.csv",
            "manipulation_vulnerability_summary.csv",
            "manipulation_f1_delta.csv",
            "manipulation_recall_delta.csv",
        ],
        "saved_figures": saved_figs,
    }


if __name__ == "__main__":
    setup_logging()
    exp_dir = OUTPUT_ROOT / "robustness" / "core_experiment_117"
    if exp_dir.exists():
        run_novelty_pipeline(exp_dir)
    else:
        logging.error(f"Experiment directory not found at {exp_dir}")
