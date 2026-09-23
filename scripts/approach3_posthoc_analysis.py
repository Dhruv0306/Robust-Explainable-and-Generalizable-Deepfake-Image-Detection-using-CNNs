"""
Post-experiment Approach 3 analysis.

This script operates only on completed video-level outputs. It does not
regenerate frames, masks, predictions, or Grad-CAM maps.

Primary statistical unit for paired explainability analyses:
    12 manipulated source-pair IDs.

The four manipulated categories sharing the same source pair are averaged
within each source pair. Original videos are excluded from quantitative
explainability analyses.

Prediction-state analysis remains at the category-video level because a
source-pair-level average cannot represent a discrete Correct/Incorrect state.

Required input:
    data/output/explainability_corrected_v1/<model>/seed_<seed>/
        video_level_results.csv

Optional Approach 2 input:
    A CSV containing normalized columns:
        model, transformation, f1_clean, f1_transformed

Usage:
    python scripts/approach3_posthoc_analysis.py \
        --root data/output/explainability_corrected_v1 \
        --out data/output/approach3_posthoc

With Approach 2 integration:
    python scripts/approach3_posthoc_analysis.py \
        --root data/output/explainability_corrected_v1 \
        --approach2-csv PATH \
        --out data/output/approach3_posthoc
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

MODELS = {
    "xception": 2024,
    "efficientnet_b0": 2024,
    "resnet50": 123,
}

FAMILIES = {
    "jpeg": ["jpeg_sev1", "jpeg_sev2", "jpeg_sev3"],
    "resize": ["resize_sev1", "resize_sev2", "resize_sev3"],
    "darkening": ["darkening_sev1", "darkening_sev2", "darkening_sev3"],
    "brightening": ["brightening_sev1", "brightening_sev2", "brightening_sev3"],
}

STABILITY_THRESHOLDS = (0.80, 0.90)


NUMERIC_SOURCE_METRICS = [
    "prob_fake",
    "correct",
    "SO",
    "IoU",
    "saliency_mass",
    "hit_rate",
    "saliency_entropy",
    "faithfulness",
    "faithfulness_blur",
    "faithfulness_zero",
    "faithfulness_mean",
    "ES_cos",
    "explanation_IoU",
]


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    p = pd.to_numeric(p_values, errors="coerce")
    result = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna()
    if not valid.any():
        return result

    values = p.loc[valid].to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)

    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adjusted[i] = running

    restored = np.empty(m, dtype=float)
    restored[order] = adjusted
    result.loc[valid] = np.clip(restored, 0.0, 1.0)
    return result


def rank_biserial_from_differences(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size from x - y differences."""
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return np.nan

    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    denom = w_plus + w_minus
    if denom == 0:
        return np.nan
    return (w_plus - w_minus) / denom


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int = 42,
    n_boot: int = 5000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def source_video(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the four manipulated category-video rows sharing a source pair
    into one source-video observation for paired quantitative analyses.

    Original rows are excluded. Binary prediction-state variables are not
    averaged for state analysis; that analysis uses the category-video data
    directly.
    """
    fake = df[df["ground_truth"] == 1].copy()
    present = [c for c in NUMERIC_SOURCE_METRICS if c in fake.columns]

    return fake.groupby(
        ["model", "seed", "video_id", "transformation"],
        as_index=False,
    )[present].mean()


def paired(
    clean: pd.DataFrame,
    transformed: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    c = clean.set_index("video_id")[metric]
    t = transformed.set_index("video_id")[metric]
    common = c.index.intersection(t.index)
    x = c.loc[common].astype(float).to_numpy()
    y = t.loc[common].astype(float).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def direct_stability_stats(source: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze ES_cos and explanation IoU as direct clean-to-transformed metrics.

    There is deliberately no clean-vs-transformed Wilcoxon test for ES_cos:
    the clean condition has no ES_cos value because ES_cos is itself computed
    between clean and transformed Grad-CAM maps.
    """
    rows = []
    for model in source["model"].unique():
        m = source[source["model"] == model]
        for cond in m["transformation"].unique():
            if cond == "clean":
                continue
            x = m[m["transformation"] == cond]
            for metric in ["ES_cos", "explanation_IoU"]:
                if metric not in x.columns:
                    continue
                vals = x[metric].dropna().to_numpy(dtype=float)
                lo, hi = bootstrap_mean_ci(vals)
                row = {
                    "model": model,
                    "transformation": cond,
                    "metric": metric,
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "median": float(np.median(vals)) if len(vals) else np.nan,
                    "ci95_lower": lo,
                    "ci95_upper": hi,
                }
                for threshold in STABILITY_THRESHOLDS:
                    label = str(threshold).replace(".", "_")
                    row[f"below_{label}_prop"] = (
                        float(np.mean(vals < threshold)) if len(vals) else np.nan
                    )
                rows.append(row)

    return pd.DataFrame(rows)


def paired_change_stats(source: pd.DataFrame) -> pd.DataFrame:
    """Paired clean-to-transformed changes at the 12 source-pair level."""
    rows = []
    metrics = ["SO", "IoU", "saliency_mass", "faithfulness_blur"]

    for model in source["model"].unique():
        m = source[source["model"] == model]
        clean = m[m["transformation"] == "clean"]
        for cond in m["transformation"].unique():
            if cond == "clean":
                continue
            trans = m[m["transformation"] == cond]
            for metric in metrics:
                if metric not in clean.columns or metric not in trans.columns:
                    continue
                x, y = paired(clean, trans, metric)
                if len(x) == 0:
                    continue

                d = x - y
                lo, hi = bootstrap_mean_ci(d)
                row = {
                    "model": model,
                    "transformation": cond,
                    "metric": metric,
                    "n": len(d),
                    "clean_mean": float(np.mean(x)),
                    "transformed_mean": float(np.mean(y)),
                    "mean_degradation": float(np.mean(d)),
                    "median_degradation": float(np.median(d)),
                    "ci95_lower": lo,
                    "ci95_upper": hi,
                    "rank_biserial": rank_biserial_from_differences(d),
                }

                nonzero = d[d != 0]
                if len(nonzero) >= 6:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = wilcoxon(nonzero)
                    row["wilcoxon_W"] = float(res.statistic)
                    row["p_raw"] = float(res.pvalue)
                else:
                    row["wilcoxon_W"] = np.nan
                    row["p_raw"] = np.nan

                rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        # FDR is applied separately for each model/metric family across the
        # 12 transformation conditions.
        out["p_bh"] = np.nan
        for (_, _), idx in out.groupby(["model", "metric"]).groups.items():
            out.loc[idx, "p_bh"] = benjamini_hochberg(out.loc[idx, "p_raw"])
    return out


def prediction_state_analysis(raw_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Preserve prediction states at category-video level.

    This avoids inventing a source-level binary state by averaging the four
    manipulation categories belonging to one source pair.
    """
    rows = []

    for model, df in raw_by_model.items():
        fake = df[df["ground_truth"] == 1].copy()
        clean = fake[fake["transformation"] == "clean"].copy()

        required = {"video_id", "category", "pred_fake", "correct"}
        missing = required - set(clean.columns)
        if missing:
            raise ValueError(
                f"{model}: prediction-state analysis missing columns: "
                + ", ".join(sorted(missing))
            )

        key = ["video_id", "category"]
        clean_idx = clean.set_index(key)

        for cond in fake["transformation"].unique():
            if cond == "clean":
                continue

            trans = fake[fake["transformation"] == cond].copy().set_index(key)
            common = clean_idx.index.intersection(trans.index)

            for idx in common:
                c = clean_idx.loc[idx]
                t = trans.loc[idx]
                clean_correct = bool(float(c["correct"]) >= 0.5)
                transformed_correct = bool(float(t["correct"]) >= 0.5)

                if clean_correct and transformed_correct:
                    state = "Correct->Correct"
                elif clean_correct and not transformed_correct:
                    state = "Correct->Incorrect"
                elif not clean_correct and transformed_correct:
                    state = "Incorrect->Correct"
                else:
                    state = "Incorrect->Incorrect"

                row = {
                    "model": model,
                    "video_id": idx[0],
                    "category": idx[1],
                    "transformation": cond,
                    "prediction_state": state,
                    "prediction_preserved": int(
                        float(c["pred_fake"]) == float(t["pred_fake"])
                    ),
                }

                for col in [
                    "SO",
                    "IoU",
                    "saliency_mass",
                    "hit_rate",
                    "faithfulness_blur",
                    "ES_cos",
                    "explanation_IoU",
                ]:
                    row[col] = t[col] if col in t.index else np.nan
                rows.append(row)

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail

    group_cols = ["model", "transformation", "prediction_state"]
    agg_cols = [
        "SO",
        "IoU",
        "saliency_mass",
        "hit_rate",
        "faithfulness_blur",
        "ES_cos",
        "explanation_IoU",
    ]

    out = (
        detail.groupby(group_cols)
        .agg(
            n=("video_id", "size"),
            **{c: (c, "mean") for c in agg_cols},
        )
        .reset_index()
    )
    totals = out.groupby(["model", "transformation"])["n"].transform("sum")
    out["proportion"] = out["n"] / totals
    return out


def severity_analysis(source: pd.DataFrame) -> pd.DataFrame:
    """Assess monotonic trends across the predefined severity orders."""
    rows = []
    for model in source["model"].unique():
        m = source[source["model"] == model]
        for family, conditions in FAMILIES.items():
            for metric in [
                "SO",
                "IoU",
                "saliency_mass",
                "faithfulness_blur",
                "ES_cos",
                "explanation_IoU",
            ]:
                means = []
                severities = []
                for severity_level, cond in enumerate(conditions, start=1):
                    x = m[m["transformation"] == cond][metric].dropna()
                    if len(x):
                        means.append(float(x.mean()))
                        # Severity is encoded by the predefined condition order
                        # in FAMILIES. The video-level output does not contain a
                        # separate `severity` column.
                        severities.append(severity_level)

                if len(means) >= 3:
                    rho, p = spearmanr(severities, means)
                    rows.append(
                        {
                            "model": model,
                            "family": family,
                            "metric": metric,
                            "severity_levels": len(means),
                            "rho": float(rho),
                            "p_raw": float(p),
                            "means": "|".join(f"{v:.6f}" for v in means),
                        }
                    )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh"] = np.nan
        for (_, _), idx in out.groupby(["model", "metric"]).groups.items():
            out.loc[idx, "p_bh"] = benjamini_hochberg(out.loc[idx, "p_raw"])
    return out


def localization_faithfulness(source: pd.DataFrame) -> pd.DataFrame:
    """Spearman association between localization and faithfulness."""
    rows = []
    for model in source["model"].unique():
        m = source[source["model"] == model]
        for cond in m["transformation"].unique():
            x = m[m["transformation"] == cond]
            for loc in ["SO", "IoU", "saliency_mass"]:
                if loc not in x.columns or "faithfulness_blur" not in x.columns:
                    continue
                valid = x[[loc, "faithfulness_blur"]].dropna()
                if len(valid) < 6:
                    continue
                rho, p = spearmanr(valid[loc], valid["faithfulness_blur"])
                rows.append(
                    {
                        "model": model,
                        "transformation": cond,
                        "localization_metric": loc,
                        "faithfulness_method": "blur",
                        "n": len(valid),
                        "rho": float(rho),
                        "p_raw": float(p),
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh"] = np.nan
        # One FDR family per model/localization metric across transformations.
        for (_, _), idx in out.groupby(["model", "localization_metric"]).groups.items():
            out.loc[idx, "p_bh"] = benjamini_hochberg(out.loc[idx, "p_raw"])
    return out


def approach2_integration(
    source: pd.DataFrame,
    a2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Integrate Approach 2 detector-performance degradation with Approach 3.

    Required normalized Approach 2 columns:
        model, transformation, f1_clean, f1_transformed

    Outputs one row per model/transformation and then Spearman relationships
    between detector-performance degradation and explanation degradation.
    """
    # Accept either the normalized bridge schema or the native Approach 2
    # master_summary.csv schema. The native schema is preferred because it is
    # the source-of-truth output of the robustness experiment.
    native_required = {
        "model",
        "seed",
        "condition_id",
        "transformation",
        "f1",
        "delta_f1",
    }
    normalized_required = {"model", "transformation", "f1_clean", "f1_transformed"}

    if native_required.issubset(a2.columns):
        # Approach 3 uses one selected seed per architecture. Restrict Approach 2
        # to the same checkpoint so the bridge compares like-for-like detectors.
        selected = []
        for model, seed in MODELS.items():
            selected.append(
                a2[
                    (a2["model"].astype(str) == str(model))
                    & (a2["seed"].astype(int) == int(seed))
                ]
            )
        a2 = pd.concat(selected, ignore_index=True)

        clean_rows = a2[a2["transformation"].astype(str) == "clean"][
            ["model", "f1"]
        ].rename(columns={"f1": "f1_clean"})
        clean_rows = clean_rows.drop_duplicates("model")

        a2 = a2[a2["transformation"].astype(str) != "clean"].copy()
        a2 = a2.merge(clean_rows, on="model", how="left", validate="many_to_one")
        a2["f1_transformed"] = a2["f1"].astype(float)

        # Native Approach 2 stores the four transformation families with a
        # family name plus a severity level. Approach 3 stores the same
        # conditions as explicit condition IDs such as jpeg_sev1. Normalize
        # the native rows to those IDs before joining the two approaches.
        family_prefix = {
            "jpeg": "jpeg_sev",
            "resize": "resize_sev",
            "brightness_dark": "darkening_sev",
            "brightness_bright": "brightening_sev",
        }
        non_clean = a2["transformation"].astype(str)
        unsupported = ~non_clean.isin(family_prefix)
        if unsupported.any():
            bad = sorted(a2.loc[unsupported, "transformation"].astype(str).unique())
            raise ValueError(
                "Unsupported native Approach 2 transformation(s): " + ", ".join(bad)
            )

        severity_num = pd.to_numeric(a2["severity"], errors="coerce")
        if severity_num.isna().any() or ~severity_num.isin([1, 2, 3]).all():
            bad = sorted(
                a2.loc[severity_num.isna() | ~severity_num.isin([1, 2, 3]), "severity"]
                .astype(str)
                .unique()
            )
            raise ValueError(
                "Native Approach 2 transformed rows must have severity 1, 2, or 3. "
                "Invalid values: " + ", ".join(bad)
            )

        a2["transformation"] = [
            f"{family_prefix[t]}{int(s)}"
            for t, s in zip(a2["transformation"].astype(str), severity_num)
        ]

        # Native delta_f1 is transformed - clean. The bridge uses positive
        # degradation, clean - transformed. Use the F1 values directly so the
        # sign convention is explicit and independent of stored delta fields.
    elif not normalized_required.issubset(a2.columns):
        missing = normalized_required - set(a2.columns)
        raise ValueError(
            "Approach 2 CSV must contain either native master_summary columns "
            "or normalized columns. Missing normalized columns: "
            + ", ".join(sorted(missing))
        )

    # Validate that the bridge contains exactly one selected-checkpoint row per
    # model/transformation. Duplicate rows would make the F1-to-explanation
    # relationship ambiguous.
    dup = a2.duplicated(["model", "transformation"], keep=False)
    if dup.any():
        counts = a2.loc[dup].groupby(["model", "transformation"]).size()
        raise ValueError(
            "Approach 2 bridge contains duplicate model/transformation rows: "
            + counts.to_string()
        )

    rows = []
    for model in source["model"].unique():
        m = source[source["model"] == model]
        clean = m[m["transformation"] == "clean"]

        for cond in m["transformation"].unique():
            if cond == "clean":
                continue

            a = a2[
                (a2["model"].astype(str) == str(model))
                & (a2["transformation"].astype(str) == str(cond))
            ]
            if a.empty:
                continue

            d_f1 = float(a.iloc[0]["f1_clean"] - a.iloc[0]["f1_transformed"])
            trans = m[m["transformation"] == cond]

            row = {
                "model": model,
                "transformation": cond,
                "D_F1": d_f1,
            }

            # Localization degradation: clean minus transformed.
            for metric, name in [("SO", "D_SO"), ("IoU", "D_IoU")]:
                c = clean[metric].dropna().mean()
                t = trans[metric].dropna().mean()
                row[name] = float(c - t)

            # Stability degradation is 1 - mean clean-to-transformed cosine
            # similarity. ES_cos is already a clean-to-transformed quantity.
            es = trans["ES_cos"].dropna().to_numpy(dtype=float)
            row["D_stability"] = 1.0 - float(np.mean(es)) if len(es) else np.nan

            # Faithfulness degradation uses the clean-transformed change in
            # the existing faithfulness metric.
            c_f = clean["faithfulness_blur"].dropna().mean()
            t_f = trans["faithfulness_blur"].dropna().mean()
            row["D_faith"] = float(c_f - t_f)

            # Preserve probability shift as a separate confidence analysis.
            c_p = clean["prob_fake"].dropna().mean()
            t_p = trans["prob_fake"].dropna().mean()
            row["D_prob_fake"] = float(c_p - t_p)

            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    corr_rows = []
    explanation_cols = ["D_SO", "D_IoU", "D_stability", "D_faith"]

    for model, g in out.groupby("model"):
        for xcol in explanation_cols:
            valid = g[["D_F1", xcol]].dropna()
            if len(valid) >= 6:
                rho, p = spearmanr(valid["D_F1"], valid[xcol])
                corr_rows.append(
                    {
                        "model": model,
                        "predictor": "D_F1",
                        "response": xcol,
                        "n": len(valid),
                        "rho": float(rho),
                        "p_raw": float(p),
                    }
                )

        # Separate confidence-shift analysis. This does not substitute for D_F1.
        for xcol in explanation_cols:
            valid = g[["D_prob_fake", xcol]].dropna()
            if len(valid) >= 6:
                rho, p = spearmanr(valid["D_prob_fake"], valid[xcol])
                corr_rows.append(
                    {
                        "model": model,
                        "predictor": "D_prob_fake",
                        "response": xcol,
                        "n": len(valid),
                        "rho": float(rho),
                        "p_raw": float(p),
                    }
                )

    correlations = pd.DataFrame(corr_rows)
    if correlations.empty:
        return out

    # Define the two predefined FDR families.
    correlations["fdr_family"] = correlations["predictor"].map(
        {
            "D_F1": "detector_performance",
            "D_prob_fake": "confidence_shift",
        }
    )

    # Apply BH-FDR separately to each family.
    #
    # detector_performance:
    #   3 models × 4 explanation responses = 12 tests
    #
    # confidence_shift:
    #   3 models × 4 explanation responses = 12 tests
    correlations["p_bh"] = np.nan

    for family, idx in correlations.groupby("fdr_family").groups.items():
        correlations.loc[idx, "p_bh"] = benjamini_hochberg(
            correlations.loc[idx, "p_raw"]
        )

    return out, correlations


def load_inputs(root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    all_source = []
    raw_by_model = {}

    for model, seed in MODELS.items():
        path = root / model / f"seed_{seed}" / "video_level_results.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path, low_memory=False)
        raw_by_model[model] = df
        all_source.append(source_video(df))

    return pd.concat(all_source, ignore_index=True), raw_by_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--approach2-csv", type=Path, default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    source, raw_by_model = load_inputs(args.root)
    source.to_csv(args.out / "source_video_results.csv", index=False)

    direct_stability_stats(source).to_csv(
        args.out / "explanation_stability.csv", index=False
    )
    paired_change_stats(source).to_csv(args.out / "paired_changes.csv", index=False)
    prediction_state_analysis(raw_by_model).to_csv(
        args.out / "prediction_state_analysis.csv", index=False
    )
    severity_analysis(source).to_csv(args.out / "severity_analysis.csv", index=False)
    localization_faithfulness(source).to_csv(
        args.out / "localization_faithfulness.csv", index=False
    )

    if args.approach2_csv:
        a2 = pd.read_csv(args.approach2_csv, low_memory=False)
        result = approach2_integration(source, a2)
        if isinstance(result, tuple):
            integration, relationships = result
            integration.to_csv(
                args.out / "approach2_explanation_integration.csv",
                index=False,
            )
            relationships.to_csv(
                args.out / "approach2_explanation_relationships.csv",
                index=False,
            )
        else:
            result.to_csv(
                args.out / "approach2_explanation_integration.csv",
                index=False,
            )

    print(f"Wrote post-hoc analysis to: {args.out}")


if __name__ == "__main__":
    main()
