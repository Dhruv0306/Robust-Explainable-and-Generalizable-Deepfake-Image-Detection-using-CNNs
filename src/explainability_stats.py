"""
Approach 3: Statistical Testing Engine & Multiple-Testing Correction.
Implements:
1. Video-level paired Wilcoxon signed-rank tests
2. Rank-biserial correlation effect size (r_rb)
3. Video-level bootstrap 95% confidence intervals (mean difference)
4. Spearman rank correlation (rho, p-value)
5. Benjamini-Hochberg False Discovery Rate (FDR) correction within 5 isolated families
6. Strict small-sample policy enforcement (min n >= 6 non-zero pairs)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats


def compute_rank_biserial_effect_size(
    diffs: np.ndarray,
    w_stat: Optional[float] = None,
) -> float:
    """
    Compute rank-biserial correlation effect size for paired Wilcoxon signed-rank test.
    Formula: r_rb = (W_+ - W_-) / (W_+ + W_-) = (4 * W_+ - n*(n+1)) / (n*(n+1))
    Range: [-1.0, 1.0].

    Args:
        diffs: 1D array of paired differences (x - y) with zeros removed
        w_stat: Optional Wilcoxon test statistic (min(W_+, W_-))

    Returns:
        float effect size in [-1.0, 1.0]
    """
    nonzero_diffs = diffs[diffs != 0]
    n = len(nonzero_diffs)
    if n == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(nonzero_diffs))
    w_plus = float(np.sum(ranks[nonzero_diffs > 0]))
    w_minus = float(np.sum(ranks[nonzero_diffs < 0]))
    total_ranks = w_plus + w_minus

    if total_ranks <= 1e-7:
        return 0.0

    return float((w_plus - w_minus) / total_ranks)


def compute_bootstrap_ci(
    values: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    stat_fn: str = "mean",
) -> Tuple[float, float]:
    """
    Compute video-level empirical bootstrap confidence interval.

    Args:
        values: 1D array of video-level observations or paired differences
        n_bootstraps: number of bootstrap repetitions (default 1000)
        alpha: significance level (0.05 for 95% CI)
        seed: random seed for reproducibility
        stat_fn: 'mean' or 'median'

    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    clean_vals = values[np.isfinite(values)]
    n = len(clean_vals)
    if n < 3:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
    resamples = clean_vals[boot_indices]

    if stat_fn == "mean":
        boot_stats = np.mean(resamples, axis=1)
    elif stat_fn == "median":
        boot_stats = np.median(resamples, axis=1)
    else:
        raise ValueError(f"Unknown stat_fn: {stat_fn}")

    lower_p = 100.0 * (alpha / 2.0)
    upper_p = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_stats, lower_p))
    ci_upper = float(np.percentile(boot_stats, upper_p))
    return ci_lower, ci_upper


def run_paired_wilcoxon_test(
    clean_values: np.ndarray,
    transformed_values: np.ndarray,
    min_nonzero_n: int = 6,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute video-level paired Wilcoxon signed-rank test with effect size and bootstrap CI.

    Args:
        clean_values: 1D array of clean video baseline values
        transformed_values: 1D array of corresponding transformed video values
        min_nonzero_n: minimum non-zero differences required (default 6)
        n_bootstraps: bootstrap repetitions (default 1000)
        seed: random seed for bootstrap

    Returns:
        Dict with test results, effect sizes, CIs, and sample size counts
    """
    clean_arr = np.asarray(clean_values, dtype=np.float64)
    trans_arr = np.asarray(transformed_values, dtype=np.float64)

    valid_mask = np.logical_and(np.isfinite(clean_arr), np.isfinite(trans_arr))
    n_total = len(clean_arr)
    n_valid = int(valid_mask.sum())

    res: Dict[str, Any] = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_nonzero": 0,
        "mean_clean": float(np.mean(clean_arr[valid_mask])) if n_valid > 0 else float("nan"),
        "mean_transformed": float(np.mean(trans_arr[valid_mask])) if n_valid > 0 else float("nan"),
        "mean_diff": float("nan"),
        "median_diff": float("nan"),
        "w_stat": float("nan"),
        "raw_p": float("nan"),
        "r_rb": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "test_status": "not_tested_small_n",
    }

    if n_valid == 0:
        return res

    diffs = clean_arr[valid_mask] - trans_arr[valid_mask]
    nonzero_diffs = diffs[diffs != 0.0]
    n_nonzero = len(nonzero_diffs)
    res["n_nonzero"] = n_nonzero
    res["mean_diff"] = float(np.mean(diffs))
    res["median_diff"] = float(np.median(diffs))

    # Bootstrap 95% CI on mean paired difference
    ci_low, ci_high = compute_bootstrap_ci(diffs, n_bootstraps=n_bootstraps, seed=seed)
    res["ci_lower"] = ci_low
    res["ci_upper"] = ci_high

    if n_nonzero < min_nonzero_n:
        res["test_status"] = "not_tested_small_n"
        return res

    try:
        w_res = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
        res["w_stat"] = float(w_res.statistic)
        res["raw_p"] = float(w_res.pvalue)
        res["r_rb"] = compute_rank_biserial_effect_size(nonzero_diffs, w_stat=res["w_stat"])
        res["test_status"] = "tested"
    except Exception as e:
        res["test_status"] = f"error: {str(e)}"

    return res


def run_spearman_correlation(
    x_values: np.ndarray,
    y_values: np.ndarray,
    min_n: int = 6,
) -> Dict[str, Any]:
    """
    Execute video-level Spearman rank correlation.

    Args:
        x_values: 1D array
        y_values: 1D array
        min_n: minimum paired observations required (default 6)

    Returns:
        Dict with rho, p-value, sample size, and test status
    """
    x_arr = np.asarray(x_values, dtype=np.float64)
    y_arr = np.asarray(y_values, dtype=np.float64)

    valid_mask = np.logical_and(np.isfinite(x_arr), np.isfinite(y_arr))
    n_total = len(x_arr)
    n_valid = int(valid_mask.sum())

    res: Dict[str, Any] = {
        "n_total": n_total,
        "n_valid": n_valid,
        "rho": float("nan"),
        "raw_p": float("nan"),
        "test_status": "not_tested_small_n",
    }

    if n_valid < min_n:
        return res

    x_valid = x_arr[valid_mask]
    y_valid = y_arr[valid_mask]

    # Check for constant variance (Spearman undefined if all identical)
    if np.all(x_valid == x_valid[0]) or np.all(y_valid == y_valid[0]):
        res["test_status"] = "not_tested_zero_variance"
        return res

    try:
        corr_res = stats.spearmanr(x_valid, y_valid)
        res["rho"] = float(corr_res.correlation) if hasattr(corr_res, "correlation") else float(corr_res.statistic)
        res["raw_p"] = float(corr_res.pvalue)
        res["test_status"] = "tested"
    except Exception as e:
        res["test_status"] = f"error: {str(e)}"

    return res


def apply_benjamini_hochberg(
    p_values: Union[List[float], np.ndarray],
) -> np.ndarray:
    """
    Apply Benjamini-Hochberg False Discovery Rate (FDR) correction.
    Formula: p_adj = min(1, min_{j >= i} (p_(j) * m / j))

    Args:
        p_values: list or 1D array of raw p-values

    Returns:
        1D numpy array of adjusted p-values of the same length
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    n = len(p_arr)
    adj_p = np.full(n, np.nan, dtype=np.float64)

    valid_mask = np.isfinite(p_arr)
    m = int(valid_mask.sum())
    if m == 0:
        return adj_p

    valid_indices = np.where(valid_mask)[0]
    sorted_order = np.argsort(p_arr[valid_mask])
    sorted_indices = valid_indices[sorted_order]
    sorted_p = p_arr[sorted_indices]

    # Step-up adjustment: cumulative min from right
    ranks = np.arange(1, m + 1)
    adjusted_sorted = (sorted_p * m) / ranks

    # Enforce monotonicity: p_adj[i] = min(p_adj[i], p_adj[i+1])
    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adj_p[sorted_indices] = adjusted_sorted
    return adj_p


def adjust_p_values_by_family(
    results_df: pd.DataFrame,
    family_col: str = "test_family",
    p_col: str = "raw_p",
    adj_p_col: str = "adjusted_p",
) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction separately within logical test families.

    Args:
        results_df: DataFrame containing test rows
        family_col: column identifying the FDR family
        p_col: column containing raw p-values
        adj_p_col: destination column name for adjusted p-values

    Returns:
        DataFrame with new/updated adj_p_col
    """
    df = results_df.copy()
    df[adj_p_col] = np.nan

    if family_col not in df.columns or p_col not in df.columns:
        return df

    for family_name, group in df.groupby(family_col):
        raw_p = group[p_col].values
        adj_p = apply_benjamini_hochberg(raw_p)
        df.loc[group.index, adj_p_col] = adj_p

    return df


if __name__ == "__main__":
    print("Testing explainability statistical engine...")

    # 1. Test paired Wilcoxon
    clean = np.array([0.85, 0.90, 0.78, 0.92, 0.88, 0.79, 0.95, 0.82])
    trans = np.array([0.65, 0.70, 0.58, 0.72, 0.68, 0.59, 0.75, 0.62])  # consistently lower
    w_res = run_paired_wilcoxon_test(clean, trans, min_nonzero_n=6)
    print(f"Wilcoxon tested: W={w_res['w_stat']}, p={w_res['raw_p']:.4f}, r_rb={w_res['r_rb']:.4f}, status={w_res['test_status']}")
    print(f"95% CI on mean diff: [{w_res['ci_lower']:.4f}, {w_res['ci_upper']:.4f}]")
    assert w_res["test_status"] == "tested"
    assert w_res["raw_p"] < 0.05
    assert w_res["r_rb"] == 1.0  # all positive differences

    # 2. Test small-sample gate
    small_clean = np.array([0.85, 0.90, 0.88])
    small_trans = np.array([0.85, 0.90, 0.88])
    w_small = run_paired_wilcoxon_test(small_clean, small_trans, min_nonzero_n=6)
    print("Small sample test status (expected 'not_tested_small_n'):", w_small["test_status"])
    assert w_small["test_status"] == "not_tested_small_n"
    assert np.isnan(w_small["raw_p"])

    # 3. Test Spearman correlation
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([2, 4, 5, 8, 10, 12, 15, 17])
    sp_res = run_spearman_correlation(x, y, min_n=6)
    print(f"Spearman tested: rho={sp_res['rho']:.4f}, p={sp_res['raw_p']:.4f}, status={sp_res['test_status']}")
    assert sp_res["test_status"] == "tested"
    assert sp_res["rho"] > 0.95

    # 4. Test Benjamini-Hochberg FDR
    raw_p = [0.001, 0.005, 0.02, 0.04, 0.05, 0.15, 0.50]
    adj_p = apply_benjamini_hochberg(raw_p)
    print("Raw p:    ", [f"{p:.3f}" for p in raw_p])
    print("Adjusted p:", [f"{p:.3f}" for p in adj_p])
    assert np.all(adj_p >= raw_p)
    assert np.all(np.diff(adj_p) >= -1e-7)  # non-decreasing

    # 5. Test family-wise adjustment
    test_df = pd.DataFrame({
        "test_family": ["family_1"] * 4 + ["family_2"] * 3,
        "raw_p": [0.01, 0.02, 0.03, 0.04, 0.005, 0.01, 0.08],
    })
    adj_df = adjust_p_values_by_family(test_df)
    print("\nFamily FDR Table:")
    print(adj_df)
    assert not adj_df["adjusted_p"].isna().any()

    print("\nAll statistical testing engine tests PASSED!")
