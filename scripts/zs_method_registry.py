#!/usr/bin/env python
"""
Zero-shot evaluation method registry.

Centralizes method configurations to eliminate duplication across:
- parallel_compare.py (METHOD_CONFIGS, METHOD_RESULT_KEYS, METHOD_STATS_KEYS)
- zs.py (run_comparison_evaluation method dispatching)
- Individual inference scripts

Usage:
    from zs_method_registry import METHOD_REGISTRY, get_method_config
    
    # Get all methods
    for name, cfg in METHOD_REGISTRY.items():
        print(f"{name}: {cfg.output_suffix}")
    
    # Get specific method
    cfg = get_method_config("BNDL_AUE")
    cfg.flags  # ["--run_bndl_aue"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class MethodConfig:
    """Configuration for a zero-shot evaluation method.
    
    Attributes:
        name: Display name (e.g., "BNDL_AUE")
        flags: CLI flags for zs.py (e.g., ["--run_bndl_aue"])
        output_suffix: Directory suffix (e.g., "bndl_aue")
        result_key: JSON key for results (e.g., "bndl_aue_results")
        stats_key: JSON key for statistics (e.g., "bndl_aue_statistics")
        color: ANSI color code for terminal output
        cfg_arg: CLI argument for config path (e.g., "--bndl_aue_cfg")
        ckpt_arg: CLI argument for checkpoint path (e.g., "--bndl_aue_checkpoint")
        uses_predictor_from: If not None, reuses predictor from specified method
        requires_sam_prompts: Whether this method requires SAM-generated prompts
        extra_args: Additional CLI arguments for this method
    """
    name: str
    flags: list[str]
    output_suffix: str
    result_key: str
    stats_key: str
    color: str = "\033[0m"  # Default: reset
    cfg_arg: str | None = None
    ckpt_arg: str | None = None
    uses_predictor_from: str | None = None
    requires_sam_prompts: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)
    
    def get_predictor_key(self) -> str:
        """Get the method name whose predictor this method uses."""
        return self.uses_predictor_from or self.name


# =============================================================================
# Method Registry
# =============================================================================

METHOD_REGISTRY: dict[str, MethodConfig] = {
    "SAM": MethodConfig(
        name="SAM",
        flags=["--run_sam"],
        output_suffix="sam",
        result_key="sam2_results",
        stats_key="sam2_statistics",
        color="\033[94m",  # Blue
        cfg_arg="--sam2_cfg",
        ckpt_arg="--sam2_checkpoint",
    ),
    "SAM_FT": MethodConfig(
        name="SAM_FT",
        flags=["--run_sam"],  # Reuses SAM's testing logic
        output_suffix="sam_ft",
        result_key="sam2_results",  # Shares result key with SAM
        stats_key="sam2_statistics",  # Shares stats key with SAM
        color="\033[34m",  # Dark Blue
        cfg_arg="--sam_ft_cfg",
        ckpt_arg="--sam_ft_checkpoint",
    ),
    "UCTTA": MethodConfig(
        name="UCTTA",
        flags=["--run_uctta"],
        output_suffix="uctta",
        result_key="uctta_results",
        stats_key="uctta_statistics",
        color="\033[92m",  # Green
        uses_predictor_from="SAM",  # Uses SAM predictor
        requires_sam_prompts=True,
        extra_args={
            "uctta_steps": 2,
            "uctta_lr": 3e-4,
            "uctta_enable_bn": True,
            "uctta_fisher_reg": True,
            "uctta_fisher_alpha": 2000.0,
            "uctta_entropy_th": 0.4,
            "uctta_selection_p": 0.1,
        },
    ),
    "BNDL_AUE": MethodConfig(
        name="BNDL_AUE",
        flags=["--run_bndl_aue"],
        output_suffix="bndl_aue",
        result_key="bndl_aue_results",
        stats_key="bndl_aue_statistics",
        color="\033[93m",  # Yellow
        cfg_arg="--bndl_aue_cfg",
        ckpt_arg="--bndl_aue_checkpoint",
        requires_sam_prompts=True,
    ),
    "BNDL": MethodConfig(
        name="BNDL",
        flags=["--run_bndl"],
        output_suffix="bndl",
        result_key="bndl_results",
        stats_key="bndl_statistics",
        color="\033[96m",  # Cyan
        cfg_arg="--bndl_cfg",
        ckpt_arg="--bndl_checkpoint",
        requires_sam_prompts=True,
    ),
    "UR-ERN": MethodConfig(
        name="UR-ERN",
        flags=["--run_ur_ern"],
        output_suffix="ur_ern",
        result_key="ur_ern_results",
        stats_key="ur_ern_statistics",
        color="\033[95m",  # Magenta
        cfg_arg="--ur_ern_cfg",
        ckpt_arg="--ur_ern_checkpoint",
        requires_sam_prompts=True,
    ),
    # SeCo-inspired uncertainty correction methods (inference-time post-processing)
    "BNDL_CORR": MethodConfig(
        name="BNDL_CORR",
        flags=["--run_bndl_corr"],
        output_suffix="bndl_corr",
        result_key="bndl_corr_results",
        stats_key="bndl_corr_statistics",
        color="\033[36m",  # Dark Cyan
        cfg_arg="--bndl_cfg",
        ckpt_arg="--bndl_checkpoint",
        requires_sam_prompts=True,
        extra_args={"enable_uncertainty_correction": True},
    ),
    "BNDL_AUE_CORR": MethodConfig(
        name="BNDL_AUE_CORR",
        flags=["--run_bndl_aue_corr"],
        output_suffix="bndl_aue_corr",
        result_key="bndl_aue_corr_results",
        stats_key="bndl_aue_corr_statistics",
        color="\033[33m",  # Dark Yellow
        cfg_arg="--bndl_aue_cfg",
        ckpt_arg="--bndl_aue_checkpoint",
        requires_sam_prompts=True,
        extra_args={"enable_uncertainty_correction": True},
    ),
    # Adversarial patch baseline (rebuttal)
    "BNDL_PATCH": MethodConfig(
        name="BNDL_PATCH",
        flags=["--run_bndl_aue"],
        output_suffix="bndl_patch",
        result_key="bndl_aue_results",
        stats_key="bndl_aue_statistics",
        color="\033[90m",  # Gray
        cfg_arg="--bndl_aue_cfg",
        ckpt_arg="--bndl_aue_checkpoint",
        requires_sam_prompts=True,
    ),
    # Non-adversarial style augmentation baselines (rebuttal)
    "BNDL_MIXSTYLE": MethodConfig(
        name="BNDL_MIXSTYLE",
        flags=["--run_bndl_aue"],  # Reuse BNDL_AUE inference (MixStyle is training-only)
        output_suffix="bndl_mixstyle",
        result_key="bndl_aue_results",
        stats_key="bndl_aue_statistics",
        color="\033[35m",  # Magenta
        cfg_arg="--bndl_aue_cfg",
        ckpt_arg="--bndl_aue_checkpoint",
        requires_sam_prompts=True,
    ),
    "BNDL_DSU": MethodConfig(
        name="BNDL_DSU",
        flags=["--run_bndl_aue"],  # Reuse BNDL_AUE inference (DSU is training-only)
        output_suffix="bndl_dsu",
        result_key="bndl_aue_results",
        stats_key="bndl_aue_statistics",
        color="\033[32m",  # Green
        cfg_arg="--bndl_aue_cfg",
        ckpt_arg="--bndl_aue_checkpoint",
        requires_sam_prompts=True,
    ),
    "BNDL_STYLEGEN": MethodConfig(
        name="BNDL_STYLEGEN",
        flags=["--run_bndl_aue"],  # Reuse BNDL_AUE inference (StyleGen is data-only)
        output_suffix="bndl_stylegen",
        result_key="bndl_aue_results",
        stats_key="bndl_aue_statistics",
        color="\033[91m",  # Light Red
        cfg_arg="--bndl_aue_cfg",
        ckpt_arg="--bndl_aue_checkpoint",
        requires_sam_prompts=True,
    ),
}

# All available methods (ordered)
ALL_METHODS = list(METHOD_REGISTRY.keys())

# ANSI color reset
RESET_COLOR = "\033[0m"


# =============================================================================
# Compatibility Helpers (for gradual migration)
# =============================================================================

def get_method_configs_dict() -> dict[str, dict[str, Any]]:
    """
    Generate METHOD_CONFIGS dict compatible with parallel_compare.py.
    
    Returns:
        Dict matching the original parallel_compare.py format:
        {
            "SAM": {
                "flags": ["--run_sam"],
                "color": "\\033[94m",
                "output_suffix": "sam",
            },
            ...
        }
    """
    return {
        name: {
            "flags": cfg.flags,
            "color": cfg.color,
            "output_suffix": cfg.output_suffix,
        }
        for name, cfg in METHOD_REGISTRY.items()
    }


def get_method_result_keys() -> dict[str, str]:
    """
    Generate METHOD_RESULT_KEYS dict compatible with parallel_compare.py.
    
    Returns:
        {"SAM": "sam2_results", "BNDL_AUE": "bndl_aue_results", ...}
    """
    return {name: cfg.result_key for name, cfg in METHOD_REGISTRY.items()}


def get_method_stats_keys() -> dict[str, str]:
    """
    Generate METHOD_STATS_KEYS dict compatible with parallel_compare.py.
    
    Returns:
        {"SAM": "sam2_statistics", "BNDL_AUE": "bndl_aue_statistics", ...}
    """
    return {name: cfg.stats_key for name, cfg in METHOD_REGISTRY.items()}


def get_method_config(method_name: str) -> MethodConfig:
    """
    Get configuration for a specific method.
    
    Args:
        method_name: Name of the method (e.g., "BNDL_AUE")
    
    Returns:
        MethodConfig for the specified method
    
    Raises:
        KeyError: If method_name is not registered
    """
    if method_name not in METHOD_REGISTRY:
        raise KeyError(
            f"Unknown method: {method_name}. "
            f"Available methods: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[method_name]


def get_methods_needing_predictor() -> dict[str, list[str]]:
    """
    Group methods by the predictor they need.
    
    Returns:
        Dict mapping predictor name to list of methods using it.
        Example: {"SAM": ["SAM", "UCTTA"], "BNDL_AUE": ["BNDL_AUE"], ...}
    """
    result: dict[str, list[str]] = {}
    for name, cfg in METHOD_REGISTRY.items():
        predictor_key = cfg.get_predictor_key()
        if predictor_key not in result:
            result[predictor_key] = []
        result[predictor_key].append(name)
    return result


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Method Registry Tests")
    print("=" * 60)
    
    print("\n1. All registered methods:")
    for name, cfg in METHOD_REGISTRY.items():
        print(f"  {name}: {cfg.flags}, output={cfg.output_suffix}")
    
    print("\n2. Compatibility dicts:")
    print(f"  METHOD_CONFIGS keys: {list(get_method_configs_dict().keys())}")
    print(f"  METHOD_RESULT_KEYS: {get_method_result_keys()}")
    print(f"  METHOD_STATS_KEYS: {get_method_stats_keys()}")
    
    print("\n3. Predictor grouping:")
    for predictor, methods in get_methods_needing_predictor().items():
        print(f"  {predictor} -> {methods}")
    
    print("\n✓ All tests passed!")
