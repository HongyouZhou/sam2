"""SeCo-inspired uncertainty correction for predicted masks.

Filters out small, high-uncertainty connected components (likely false positives)
while always preserving the largest component (main prediction).

Reference: SeCoV2 (Dong Zhao et al., PAMI'25) — adapted from
connectivity-level pseudo-label filtering to uncertainty-guided
mask correction for SAM2+BNDL/RUAC.
"""

import logging

import numpy as np
from skimage.measure import label as connected_components

logger = logging.getLogger(__name__)


def apply_uncertainty_correction(
    binary_mask: np.ndarray,
    uncertainty: np.ndarray,
    component_threshold: float = 0.5,
    min_component_area: int = 0,
) -> tuple[np.ndarray, dict]:
    """Apply uncertainty-based connected component filtering.

    Strategy: always keep the largest connected component (main prediction).
    For smaller components, remove those with mean uncertainty above an
    adaptive threshold: max(P95 of foreground uncertainty, 0.3). The absolute
    floor prevents over-filtering on datasets where the model is overall
    confident (low P95 → low threshold → too many removals).

    Args:
        binary_mask: Boolean predicted mask [H, W].
        uncertainty: Pixel-level uncertainty map [H, W], values in [0, 1].
        component_threshold: Unused (kept for API compatibility).
        min_component_area: Always remove components smaller than this (pixels).

    Returns:
        corrected_mask: Boolean mask after correction [H, W].
        metadata: Dict with correction statistics.
    """
    if not binary_mask.any():
        return binary_mask.copy(), {"n_components": 0, "n_removed": 0, "n_kept": 0, "components": []}

    # Resize uncertainty to match mask if shapes differ
    if uncertainty.shape != binary_mask.shape:
        from skimage.transform import resize

        uncertainty = resize(uncertainty, binary_mask.shape, order=1, preserve_range=True).astype(np.float32)

    # Step 1: Connected component labeling (8-connectivity)
    labeled, n_components = connected_components(binary_mask.astype(np.uint8), return_num=True, connectivity=2)

    # Only 1 component → nothing to filter, keep as-is
    if n_components <= 1:
        return binary_mask.copy(), {
            "n_components": n_components,
            "n_removed": 0,
            "n_kept": n_components,
            "components": [],
            "skipped": True,
            "skip_reason": "single_component",
        }

    # Compute per-component stats
    comp_info = []
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        comp_area = int(comp_mask.sum())
        comp_mean_unc = float(uncertainty[comp_mask].mean())
        comp_info.append({"id": comp_id, "mask": comp_mask, "area": comp_area, "mean_unc": comp_mean_unc})

    # Find the largest component (always kept)
    largest = max(comp_info, key=lambda c: c["area"])

    # Adaptive threshold with absolute floor to prevent over-filtering
    # on low-uncertainty datasets (e.g., BBBC038v1 P95=0.068 → floor=0.3)
    MIN_COMPONENT_THRESH = 0.3
    fg_uncertainty = uncertainty[binary_mask > 0]
    adaptive_thresh = max(float(np.percentile(fg_uncertainty, 95)), MIN_COMPONENT_THRESH)

    corrected = np.zeros_like(binary_mask)
    n_removed = 0
    component_stats = []

    for comp in comp_info:
        keep = True
        reason = "kept"

        if comp["id"] == largest["id"]:
            reason = "largest_component"
        elif min_component_area > 0 and comp["area"] < min_component_area:
            keep = False
            reason = f"too_small({comp['area']})"
        elif comp["mean_unc"] > adaptive_thresh:
            keep = False
            reason = f"high_uncertainty({comp['mean_unc']:.3f}>{adaptive_thresh:.3f})"

        if keep:
            corrected[comp["mask"]] = True
        else:
            n_removed += 1

        component_stats.append({
            "id": comp["id"],
            "area": comp["area"],
            "mean_unc": comp["mean_unc"],
            "keep": keep,
            "reason": reason,
        })

    metadata = {
        "n_components": n_components,
        "n_removed": n_removed,
        "n_kept": n_components - n_removed,
        "adaptive_threshold": adaptive_thresh,
        "largest_component_id": largest["id"],
        "largest_component_area": largest["area"],
        "fg_uncertainty_mean": float(fg_uncertainty.mean()),
        "components": component_stats,
    }
    return corrected, metadata
