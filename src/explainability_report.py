"""
Approach 3: Research Report Generator.

Generates the final analysis tables and visual figures needed for the research
report. Reads the incremental CSV outputs produced by ExplainabilityOrchestrator
and produces:

1. Per-condition summary statistics (SO, IoU, faithfulness, stability, prediction states)
2. Global and per-category Grad-CAM heatmaps
3. Representative case panels (highest/lowest SO, correct/missed fake)
4. Cross-architecture comparison tables
5. The main summary JSON with all metrics ready for report pasting

Usage:
    python src/explainability_report.py --model xception --pilot

Or without --pilot for full production mode.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2

from explainability_metrics import (
    evaluate_frame_localization,
    evaluate_frame_stability,
    compute_saliency_mass,
    compute_max_activation_hit,
    compute_saliency_entropy,
)
from explainability_visuals import generate_global_heatmaps, generate_representative_cases
from utils import get_device

logger = logging.getLogger("approach3.report")


def load_frame_dataframe(model_name: str, seed: int, pilot_mode: bool = False) -> pd.DataFrame:
    """Load the frame-level results CSV for a given model/seed."""
    base = Path("data/output/explainability") / model_name / f"seed_{seed}"
    csv_path = base / "frame_level_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Frame CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if pilot_mode:
        # Pilot mode samples 5 frames per video across 1-2 videos per category
        df = df.groupby("video_id").head(5).copy()
    return df


def load_video_dataframe(model_name: str, seed: int, pilot_mode: bool = False) -> pd.DataFrame:
    """Load the video-level results CSV."""
    base = Path("data/output/explainability") / model_name / f"seed_{seed}"
    csv_path = base / "video_level_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Video CSV not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


def load_statistics_dataframe(model_name: str, seed: int) -> pd.DataFrame:
    """Load the statistical test results. Returns empty DataFrame if file is empty."""
    base = Path("data/output/explainability") / model_name / f"seed_{seed}"
    csv_path = base / "statistics_results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return pd.DataFrame()
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def generate_report_tables(model_name: str, seed: int, pilot_mode: bool = False) -> Dict[str, Any]:
    """
    Generate the main analysis tables for the final report.

    Returns a dict with table-ready data structures.
    """
    df = load_frame_dataframe(model_name, seed, pilot_mode)
    vdf = load_video_dataframe(model_name, seed, pilot_mode)
    sdf = load_statistics_dataframe(model_name, seed)

    # ---- 1. Clean vs transformed local metrics ----
    clean = vdf[vdf["transformation"] == "clean"]
    tables: Dict[str, Any] = {}

    # Per-condition SO/IoU/Saliency Mass/HitRate
    tables["localization"] = {
        "condition": [],
        "mean_SO": [],
        "mean_IoU": [],
        "mean_SM": [],
        "mean_Hit": [],
        "std_SO": [],
        "std_IoU": [],
        "std_SM": [],
        "std_Hit": [],
    }

    for cond in vdf["transformation"].unique():
        sub = vdf[vdf["transformation"] == cond]
        if sub.empty:
            continue
        so_vals = sub["SO"].dropna() if "SO" in sub.columns else pd.Series(dtype=float)
        iou_vals = sub["IoU"].dropna() if "IoU" in sub.columns else pd.Series(dtype=float)
        sm_vals = sub["saliency_mass"].dropna() if "saliency_mass" in sub.columns else pd.Series(dtype=float)
        hit_vals = sub["hit_rate"].dropna() if "hit_rate" in sub.columns else pd.Series(dtype=float)
        tables["localization"]["condition"].append(str(cond))
        tables["localization"]["mean_SO"].append(float(so_vals.mean()) if len(so_vals) else np.nan)
        tables["localization"]["mean_IoU"].append(float(iou_vals.mean()) if len(iou_vals) else np.nan)
        tables["localization"]["mean_SM"].append(float(sm_vals.mean()) if len(sm_vals) else np.nan)
        tables["localization"]["mean_Hit"].append(float(hit_vals.mean()) if len(hit_vals) else np.nan)
        tables["localization"]["std_SO"].append(float(so_vals.std()) if len(so_vals) > 1 else np.nan)
        tables["localization"]["std_IoU"].append(float(iou_vals.std()) if len(iou_vals) > 1 else np.nan)
        tables["localization"]["std_SM"].append(float(sm_vals.std()) if len(sm_vals) > 1 else np.nan)
        tables["localization"]["std_Hit"].append(float(hit_vals.std()) if len(hit_vals) > 1 else np.nan)

    # ---- 2. Faithfulness by masking method ----
    methods = [("faithfulness_blur", "blur"), ("faithfulness_zero", "zero"), ("faithfulness_mean", "mean")]
    tables["faithfulness"] = {
        "condition": [],
        "mean_delta_blur": [],
        "mean_delta_zero": [],
        "mean_delta_mean": [],
        "std_delta_blur": [],
        "std_delta_zero": [],
        "std_delta_mean": [],
    }
    for cond in vdf["transformation"].unique():
        sub = vdf[vdf["transformation"] == cond]
        if sub.empty:
            continue
        tables["faithfulness"]["condition"].append(str(cond))
        for col, tag in methods:
            vals = sub[col].dropna() if col in sub.columns else pd.Series(dtype=float)
            tables["faithfulness"][f"mean_delta_{tag}"].append(float(vals.mean()) if len(vals) else np.nan)
            tables["faithfulness"][f"std_delta_{tag}"].append(float(vals.std()) if len(vals) > 1 else np.nan)

    # ---- 3. Explanation stability (ES_cos, IoU_exp) ----
    tables["stability"] = {
        "condition": [],
        "mean_ES_cos": [],
        "mean_ES_cos_std": [],
        "mean_IoU_exp": [],
        "mean_IoU_exp_std": [],
    }
    for cond in vdf["transformation"].unique():
        sub = vdf[vdf["transformation"] == cond]
        if sub.empty:
            continue
        tables["stability"]["condition"].append(str(cond))
        cos_vals = sub["ES_cos"].dropna() if "ES_cos" in sub.columns else pd.Series(dtype=float)
        iou_vals = sub["explanation_IoU"].dropna() if "explanation_IoU" in sub.columns else pd.Series(dtype=float)
        tables["stability"]["mean_ES_cos"].append(float(cos_vals.mean()) if len(cos_vals) else np.nan)
        tables["stability"]["mean_ES_cos_std"].append(float(cos_vals.std()) if len(cos_vals) > 1 else np.nan)
        tables["stability"]["mean_IoU_exp"].append(float(iou_vals.mean()) if len(iou_vals) else np.nan)
        tables["stability"]["mean_IoU_exp_std"].append(float(iou_vals.std()) if len(iou_vals) > 1 else np.nan)

    # ---- 4. Prediction-state breakdown ----
    tables["prediction_states"] = {
        "state": [],
        "count": [],
        "mean_SO": [],
        "mean_IoU": [],
        "mean_faithfulness": [],
        "mean_ES_cos": [],
    }
    for state_label in ["Correct->Correct", "Correct->Incorrect", "Incorrect->Correct", "Incorrect->Incorrect"]:
        sub = vdf[vdf["prediction_state"] == state_label]
        if sub.empty:
            continue
        tables["prediction_states"]["state"].append(str(state_label))
        tables["prediction_states"]["count"].append(int(len(sub)))
        if "SO" in sub.columns:
            tables["prediction_states"]["mean_SO"].append(float(sub["SO"].mean()))
        else:
            tables["prediction_states"]["mean_SO"].append(np.nan)
        if "IoU" in sub.columns:
            tables["prediction_states"]["mean_IoU"].append(float(sub["IoU"].mean()))
        else:
            tables["prediction_states"]["mean_IoU"].append(np.nan)
        if "faithfulness" in sub.columns:
            tables["prediction_states"]["mean_faithfulness"].append(float(sub["faithfulness"].mean()))
        else:
            tables["prediction_states"]["mean_faithfulness"].append(np.nan)
        if "ES_cos" in sub.columns:
            tables["prediction_states"]["mean_ES_cos"].append(float(sub["ES_cos"].mean()))
        else:
            tables["prediction_states"]["mean_ES_cos"].append(np.nan)

    # ---- 5. Summary metadata ----
    tables["summary"] = {
        "model": model_name,
        "seed": int(seed),
        "pilot_mode": bool(pilot_mode),
        "total_frames": int(len(df)),
        "total_videos": int(vdf["video_id"].nunique()),
        "conditions_evaluated": vdf["transformation"].unique().tolist(),
        "n_valid_frames": int(vdf["n_valid_frames"].sum()) if "n_valid_frames" in vdf else len(df),
        "n_failed_frames": int(vdf["n_failed_frames"].sum()) if "n_failed_frames" in vdf else 0,
    }

    return tables


def write_report_tables(model_name: str, seed: int, pilot_mode: bool = False, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Write the report tables as CSV files ready for pasting into the research paper.

    Returns a dict of output file paths.
    """
    tables = generate_report_tables(model_name, seed, pilot_mode)
    out = Path(output_dir) if output_dir else Path("data/output/report_tables")
    out.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    # Localization summary
    lo = tables["localization"]
    lo_df = pd.DataFrame(lo)
    lo_path = out / "localization_summary.csv"
    lo_df.to_csv(lo_path, index=False)
    written["localization_summary"] = lo_path

    # Faithfulness summary
    fa = tables["faithfulness"]
    fa_df = pd.DataFrame(fa)
    fa_path = out / "faithfulness_summary.csv"
    fa_df.to_csv(fa_path, index=False)
    written["faithfulness_summary"] = fa_path

    # Stability summary
    st = tables["stability"]
    st_df = pd.DataFrame(st)
    st_path = out / "stability_summary.csv"
    st_df.to_csv(st_path, index=False)
    written["stability_summary"] = st_path

    # Prediction states
    ps = tables["prediction_states"]
    ps_df = pd.DataFrame(ps)
    ps_path = out / "prediction_states_summary.csv"
    ps_df.to_csv(ps_path, index=False)
    written["prediction_states_summary"] = ps_path

    # Summary metadata
    sm = tables["summary"]
    summary_path = out / "summary.json"
    with open(ps_path.parent / "report_summary.json", "w") as f:
        json.dump(sm, f, indent=2)
    written["summary_json"] = summary_path

    return written


def generate_all_figures(model_name: str, seed: int, pilot_mode: bool = False) -> Dict[str, Path]:
    """
    Generate all report figures: global heatmaps and representative case panels.

    Returns dict of output file paths.
    """
    base = Path("data/output/explainability") / model_name / f"seed_{seed}"
    cache_dir = base / "cache" / "gradcam"
    figs_dir = base / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = load_frame_dataframe(model_name, seed, pilot_mode)

    written: Dict[str, Path] = {}

    # Global heatmaps (overall + per category)
    hm_paths = generate_global_heatmaps(df, cache_dir, figs_dir)
    for k, v in hm_paths.items():
        written[f"global_heatmap_{k}"] = v

    # Representative case panels
    case_paths = generate_representative_cases(df, cache_dir, figs_dir)
    for p in case_paths:
        written[f"case_{Path(p).stem}"] = p

    return written


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Approach 3 Report Generator")
    parser.add_argument("--model", type=str, default="xception", choices=["xception", "efficientnet_b0", "resnet50"])
    parser.add_argument("--seed", type=int, default=None, help="Manual seed override")
    parser.add_argument("--pilot", action="store_true", help="Run in pilot mode (5 frames per video)")
    parser.add_argument("--tables-only", action="store_true", help="Generate only CSV tables, no figures")
    args = parser.parse_args()

    # Auto-select seed if not provided
    if args.seed is None:
        from seed_selector import select_best_seed
        from config import OUTPUT_ROOT, CHECKPOINT_ROOT
        run_info = select_best_seed(args.model, OUTPUT_ROOT, CHECKPOINT_ROOT)
        seed = int(run_info["seed"])
        print(f"Auto-selected seed: {seed}")
    else:
        seed = args.seed

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(f"Generating Approach 3 report for {args.model} (seed={seed}, pilot={args.pilot})")

    # 1. Report tables
    tables = generate_report_tables(args.model, seed, args.pilot)
    written = write_report_tables(args.model, seed, args.pilot)
    print("Report tables written:")
    for k, v in written.items():
        print(f"  {k}: {v}")

    # 2. Figures
    if not args.tables_only:
        figs = generate_all_figures(args.model, seed, args.pilot)
        print("Figures written:")
        for k, v in figs.items():
            print(f"  {k}: {v}")

    print("Report generation complete.")