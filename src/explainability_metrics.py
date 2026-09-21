r"""
Approach 3: Pure Explainability Metrics Engine & Faithfulness Interventions.
Implements:
1. Saliency thresholding (top 10%, 20%, 30% activation percentiles)
2. Saliency Overlap (SO = |S \cap M| / |S|)
3. Regional IoU (IoU = |S \cap M| / |S \cup M|)
4. Continuous Saliency Mass (SM_mask = sum_{p \in M} G(p) / sum_p G(p))
5. Maximum-Activation Mask Hit Rate (Hit = 1 if argmax(G) in M else 0)
6. Saliency Entropy (Shannon entropy of normalized distribution)
7. Explanation Stability:
   - Continuous Cosine Similarity (ES_cos = (G_c . G_t) / (||G_c|| ||G_t||))
   - Binary Explanation IoU (IoU_exp = |S_c \cap S_t| / |S_c \cup S_t|)
8. Faithfulness image masking (blur, zero, mean) applied before normalization
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


def get_salient_mask(cam_map: np.ndarray, top_fraction: float = 0.20) -> np.ndarray:
    """
    Extract binary salient region corresponding to top X% activations.

    Args:
        cam_map: float32 array in [0.0, 1.0] of shape (H, W)
        top_fraction: fraction in (0.0, 1.0], e.g. 0.20 for top 20%

    Returns:
        Binary uint8 array of shape (H, W) where 1 indicates salient pixel
    """
    if cam_map.size == 0 or cam_map.max() <= 1e-7:
        return np.zeros_like(cam_map, dtype=np.uint8)

    k = max(1, int(round(top_fraction * cam_map.size)))
    flat = cam_map.flatten()

    # Identify the top-k highest activation values
    top_indices = np.argpartition(flat, -k)[-k:]
    threshold_val = float(flat[top_indices].min())

    if threshold_val <= 0.0:
        # If threshold hits zero, only include strictly positive activations in top-k
        salient_mask = np.zeros_like(flat, dtype=np.uint8)
        positive_top = [idx for idx in top_indices if flat[idx] > 0.0]
        salient_mask[positive_top] = 1
        return salient_mask.reshape(cam_map.shape)

    return (cam_map >= threshold_val).astype(np.uint8)


def compute_saliency_overlap(salient_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    r"""
    Compute Saliency Overlap (SO = |S \cap M| / |S|).
    Measures the fraction of salient pixels lying inside the manipulated region.

    Args:
        salient_mask: binary uint8 array (H, W)
        ground_truth_mask: binary uint8 array (H, W)

    Returns:
        float in [0.0, 1.0]
    """
    s_sum = int(salient_mask.sum())
    if s_sum == 0:
        return 0.0

    intersection = int(np.logical_and(salient_mask == 1, ground_truth_mask == 1).sum())
    return float(intersection / s_sum)


def compute_regional_iou(salient_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    r"""
    Compute Regional IoU (IoU = |S \cap M| / |S \cup M|).
    Measures spatial Jaccard agreement between salient region and manipulation mask.

    Args:
        salient_mask: binary uint8 array (H, W)
        ground_truth_mask: binary uint8 array (H, W)

    Returns:
        float in [0.0, 1.0]
    """
    intersection = int(np.logical_and(salient_mask == 1, ground_truth_mask == 1).sum())
    union = int(np.logical_or(salient_mask == 1, ground_truth_mask == 1).sum())
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_saliency_mass(cam_map: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    r"""
    Compute Saliency Mass inside manipulation mask:
    SM_mask = sum_{p \in M} G(p) / sum_p G(p)

    Args:
        cam_map: continuous float32 array in [0.0, 1.0]
        ground_truth_mask: binary uint8 array (H, W)

    Returns:
        float in [0.0, 1.0]
    """
    total_mass = float(cam_map.sum())
    if total_mass <= 1e-7:
        return 0.0

    inside_mass = float(cam_map[ground_truth_mask == 1].sum())
    return float(inside_mass / total_mass)


def compute_max_activation_hit(cam_map: np.ndarray, ground_truth_mask: np.ndarray) -> int:
    """
    Compute Maximum-Activation Mask Hit Rate:
    Hit = 1 if argmax(G) is inside M, else 0.

    Args:
        cam_map: float32 array (H, W)
        ground_truth_mask: binary uint8 array (H, W)

    Returns:
        1 if hit else 0
    """
    if cam_map.size == 0 or cam_map.max() <= 1e-7:
        return 0

    idx = np.unravel_index(np.argmax(cam_map), cam_map.shape)
    return int(ground_truth_mask[idx] == 1)


def compute_saliency_entropy(cam_map: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Shannon entropy of the normalized continuous Grad-CAM map:
    H(G) = - sum p(x) * log2(p(x))

    Args:
        cam_map: float32 array (H, W)
        eps: small numerical stability constant

    Returns:
        float entropy in bits
    """
    total = float(cam_map.sum())
    if total <= 1e-7:
        return 0.0

    prob_dist = (cam_map / total).flatten()
    prob_dist = prob_dist[prob_dist > eps]
    return float(-np.sum(prob_dist * np.log2(prob_dist)))


def compute_explanation_cosine_similarity(cam_clean: np.ndarray, cam_transformed: np.ndarray) -> float:
    """
    Compute Explanation Stability via Cosine Similarity:
    ES_cos = (G_c . G_t) / (||G_c|| ||G_t||)

    Args:
        cam_clean: float32 array (H, W)
        cam_transformed: float32 array (H, W)

    Returns:
        float in [0.0, 1.0]
    """
    vec_c = cam_clean.flatten().astype(np.float64)
    vec_t = cam_transformed.flatten().astype(np.float64)

    norm_c = np.linalg.norm(vec_c)
    norm_t = np.linalg.norm(vec_t)

    if norm_c <= 1e-7 or norm_t <= 1e-7:
        # Both zero means identical null maps; one zero means completely divergent
        return 1.0 if (norm_c <= 1e-7 and norm_t <= 1e-7) else 0.0

    cos_sim = float(np.dot(vec_c, vec_t) / (norm_c * norm_t))
    return float(np.clip(cos_sim, 0.0, 1.0))


def compute_explanation_iou(
    salient_clean: np.ndarray,
    salient_transformed: np.ndarray,
) -> float:
    r"""
    Compute Explanation Stability via Binary Saliency IoU:
    IoU_exp = |S_c \cap S_t| / |S_c \cup S_t|

    Args:
        salient_clean: binary uint8 array (H, W)
        salient_transformed: binary uint8 array (H, W)

    Returns:
        float in [0.0, 1.0]
    """
    intersection = int(np.logical_and(salient_clean == 1, salient_transformed == 1).sum())
    union = int(np.logical_or(salient_clean == 1, salient_transformed == 1).sum())
    if union == 0:
        return 1.0 if (salient_clean.sum() == 0 and salient_transformed.sum() == 0) else 0.0
    return float(intersection / union)


def mask_salient_region(
    image_rgb: np.ndarray,
    salient_mask: np.ndarray,
    method: str = "blur",
    blur_kernel_size: int = 25,
) -> np.ndarray:
    """
    Apply intervention masking to the salient region of a face crop.
    Optimized for low-memory footprint and zero-allocation sub-region blurring.

    Args:
        image_rgb: uint8 RGB numpy array (H, W, 3)
        salient_mask: binary uint8 array (H, W) where 1 indicates region to mask
        method: 'blur' (primary), 'zero', or 'mean'
        blur_kernel_size: odd integer for Gaussian blur kernel (default 25)

    Returns:
        Masked uint8 RGB numpy array (H, W, 3)
    """
    if method not in ["blur", "zero", "mean"]:
        raise ValueError(f"Unknown masking method: {method}. Choose from ['blur', 'zero', 'mean']")

    masked = image_rgb.copy()
    mask_indices = (salient_mask == 1)

    if not np.any(mask_indices):
        return masked

    if method == "blur":
        # Option 2 Optimization: Blur ONLY the bounding box around the salient mask
        # Avoids full-image GaussianBlur allocations on large (1080x970) crops
        rows = np.any(mask_indices, axis=1)
        cols = np.any(mask_indices, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Pad bounding box slightly to avoid boundary artifact
        pad = blur_kernel_size
        h, w = image_rgb.shape[:2]
        rmin = max(0, rmin - pad)
        rmax = min(h, rmax + pad + 1)
        cmin = max(0, cmin - pad)
        cmax = min(w, cmax + pad + 1)

        sub_region = image_rgb[rmin:rmax, cmin:cmax]
        ksize = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        blurred_sub = cv2.GaussianBlur(sub_region, (ksize, ksize), sigmaX=0)

        # Apply blurred pixels only to mask inside sub-region
        sub_mask = mask_indices[rmin:rmax, cmin:cmax]
        masked_sub = masked[rmin:rmax, cmin:cmax]
        masked_sub[sub_mask] = blurred_sub[sub_mask]
        masked[rmin:rmax, cmin:cmax] = masked_sub

    elif method == "zero":
        masked[mask_indices] = 0

    elif method == "mean":
        unmasked = image_rgb[~mask_indices]
        if len(unmasked) > 0:
            channel_mean = unmasked.mean(axis=0).astype(np.uint8)
        else:
            channel_mean = image_rgb.mean(axis=(0, 1)).astype(np.uint8)
        masked[mask_indices] = channel_mean

    return masked


def evaluate_frame_localization(
    cam_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    thresholds: List[float] = [0.10, 0.20, 0.30],
) -> Dict[str, Any]:
    """
    Comprehensive localization evaluation for a single frame.

    Returns dict containing:
    - SO@10, SO@20 (primary), SO@30
    - IoU@10, IoU@20 (primary), IoU@30
    - saliency_mass
    - hit_rate
    - saliency_entropy
    """
    results: Dict[str, Any] = {
        "saliency_mass": compute_saliency_mass(cam_map, ground_truth_mask),
        "hit_rate": compute_max_activation_hit(cam_map, ground_truth_mask),
        "saliency_entropy": compute_saliency_entropy(cam_map),
    }

    for frac in thresholds:
        pct = int(frac * 100)
        s_mask = get_salient_mask(cam_map, top_fraction=frac)
        results[f"SO@{pct}"] = compute_saliency_overlap(s_mask, ground_truth_mask)
        results[f"IoU@{pct}"] = compute_regional_iou(s_mask, ground_truth_mask)

    # Primary aliases for convenience
    results["SO"] = results["SO@20"]
    results["IoU"] = results["IoU@20"]
    return results


def evaluate_frame_stability(
    cam_clean: np.ndarray,
    cam_transformed: np.ndarray,
    thresholds: List[float] = [0.10, 0.20, 0.30],
) -> Dict[str, Any]:
    """
    Comprehensive explanation stability evaluation between clean and transformed CAMs.
    Defensively matches spatial resolutions if clean and transformed maps differ.

    Returns dict containing:
    - ES_cos (continuous cosine similarity)
    - explanation_IoU@10, explanation_IoU@20 (primary), explanation_IoU@30
    """
    # Defensive spatial alignment: match clean map to transformed map shape
    c_clean = cam_clean
    if cam_clean.shape != cam_transformed.shape:
        th, tw = cam_transformed.shape[:2]
        c_clean = cv2.resize(cam_clean.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
        c_clean = np.clip(c_clean, 0.0, 1.0)

    results: Dict[str, Any] = {
        "ES_cos": compute_explanation_cosine_similarity(c_clean, cam_transformed),
    }

    for frac in thresholds:
        pct = int(frac * 100)
        sc = get_salient_mask(c_clean, top_fraction=frac)
        st = get_salient_mask(cam_transformed, top_fraction=frac)
        results[f"explanation_IoU@{pct}"] = compute_explanation_iou(sc, st)

    # Primary alias
    results["explanation_IoU"] = results["explanation_IoU@20"]
    return results


if __name__ == "__main__":
    # Self-test unit verification
    print("Testing explainability metrics...")

    # 1. Test masks
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:60, 20:60] = 1  # 40x40 = 1600 pixels

    cam = np.zeros((100, 100), dtype=np.float32)
    cam[20:40, 20:40] = 1.0  # 20x20 = 400 pixels inside GT

    so = compute_saliency_overlap(get_salient_mask(cam, top_fraction=0.20), gt)
    print("SO test (expected ~1.0):", so)
    assert so >= 0.99, f"Expected SO ~ 1.0, got {so}"

    iou = compute_regional_iou(get_salient_mask(cam, top_fraction=0.20), gt)
    print("IoU test (expected > 0.0):", iou)

    sm = compute_saliency_mass(cam, gt)
    print("Saliency Mass (expected 1.0):", sm)
    assert abs(sm - 1.0) < 1e-5, f"Expected SM 1.0, got {sm}"

    hit = compute_max_activation_hit(cam, gt)
    print("Max-activation hit (expected 1):", hit)
    assert hit == 1

    ent = compute_saliency_entropy(cam)
    print("Saliency entropy:", ent)

    # 2. Test stability
    cam2 = cam * 0.9 + np.random.uniform(0, 0.1, cam.shape).astype(np.float32)
    cam2 = np.clip(cam2, 0.0, 1.0)
    cos_sim = compute_explanation_cosine_similarity(cam, cam2)
    print("Cosine similarity (expected > 0.9):", cos_sim)
    assert cos_sim > 0.90, f"Expected cos_sim > 0.90, got {cos_sim}"

    # 3. Test masking
    dummy_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    s_mask = np.zeros((100, 100), dtype=np.uint8)
    s_mask[40:60, 40:60] = 1

    b_masked = mask_salient_region(dummy_img, s_mask, method="blur")
    z_masked = mask_salient_region(dummy_img, s_mask, method="zero")
    m_masked = mask_salient_region(dummy_img, s_mask, method="mean")

    assert z_masked[50, 50].sum() == 0, "Zero masking failed!"
    print("Zero masking verified.")
    assert b_masked.shape == dummy_img.shape
    assert m_masked.shape == dummy_img.shape

    print("All explainability metrics passed self-test!")
