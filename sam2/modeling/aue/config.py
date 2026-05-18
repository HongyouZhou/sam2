# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AUE Configuration Dataclasses.

Provides structured configuration management for Adversarial Uncertainty Estimation (AUE).
Replaces the 60+ flat parameters in SAM2Base.__init__ with hierarchical dataclasses.

Usage:
    # From existing flat YAML config (backward compatible)
    aue_cfg = AUEConfig.from_model_kwargs(**model_kwargs)

    # Direct construction
    aue_cfg = AUEConfig(
        enabled=True,
        style=StyleAdvConfig(enabled=True, epsilon=1.5),
        deform=DeformAdvConfig(enabled=True, epsilon=3.0),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# =============================================================================
# Style Adversarial Configuration
# =============================================================================
@dataclass
class StyleGCNConfig:
    """GCN-based multi-object style refinement configuration."""

    enabled: bool = False
    hidden_dim: int = 64
    num_layers: int = 2
    edge_threshold: float = 0.0
    use_semantic_edges: bool = True
    use_background_edges: bool = True
    distance_threshold: float = 1.0
    use_boundary_distance: bool = True
    use_visual_features: bool = True
    feature_dim: int = 256
    feature_sim_threshold: float = 0.5


@dataclass
class StyleAdvConfig:
    """Style adversarial attack configuration."""

    enabled: bool = False
    mode: Literal["image_level", "feature_level"] = "image_level"
    epsilon: float = 1.1  # Perturbation budget for style
    use_gt_region_style: bool = True

    # Global-Local Mixed Style
    use_global_local_mix: bool = False
    global_epsilon: float = 0.5
    global_weight: float = 0.5

    # GCN sub-config
    gcn: StyleGCNConfig = field(default_factory=StyleGCNConfig)

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> StyleAdvConfig:
        """Construct from flat config dict (backward compatible)."""
        gcn = StyleGCNConfig(
            enabled=d.get("style_adv_use_gcn", False),
            hidden_dim=d.get("style_adv_gcn_hidden_dim", 64),
            num_layers=d.get("style_adv_gcn_num_layers", 2),
            edge_threshold=d.get("style_adv_gcn_edge_threshold", 0.0),
            use_semantic_edges=d.get("style_adv_gcn_use_semantic_edges", True),
            use_background_edges=d.get("style_adv_gcn_use_background_edges", True),
            distance_threshold=d.get("style_adv_gcn_distance_threshold", 1.0),
            use_boundary_distance=d.get("style_adv_gcn_use_boundary_distance", True),
            use_visual_features=d.get("style_adv_gcn_use_visual_features", True),
            feature_dim=d.get("style_adv_gcn_feature_dim", 256),
            feature_sim_threshold=d.get("style_adv_gcn_feature_sim_threshold", 0.5),
        )
        return cls(
            enabled=d.get("use_style_adv", False),
            mode=d.get("style_adv_mode", "image_level"),
            epsilon=d.get("style_adv_epsilon", 1.1),
            use_gt_region_style=d.get("style_adv_use_gt_region_style", True),
            use_global_local_mix=d.get("style_adv_use_global_local_mix", False),
            global_epsilon=d.get("style_adv_global_epsilon", 0.5),
            global_weight=d.get("style_adv_global_weight", 0.5),
            gcn=gcn,
        )


# =============================================================================
# Deformation Adversarial Configuration
# =============================================================================
@dataclass
class DeformAdvConfig:
    """Deformation adversarial attack configuration."""

    enabled: bool = False
    epsilon: float = 3.0  # Deformation strength in pixels
    use_soft_composite: bool = True
    temperature: float = 1.0

    # GCN coordination
    use_gcn: bool = False
    gcn_num_layers: int = 2
    num_deform_groups: int = 4

    # Feature-based deformation options
    init_from_memory_encoder: bool = True
    freeze_encoder_components: bool = True
    zero_mean_offsets: bool = True
    local_offset_gain: float = 1.3

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> DeformAdvConfig:
        """Construct from flat config dict (backward compatible)."""
        return cls(
            enabled=d.get("use_deform_adv", False),
            epsilon=d.get("deform_adv_epsilon", 3.0),
            use_soft_composite=d.get("deform_adv_use_soft_composite", True),
            temperature=d.get("deform_adv_temperature", 1.0),
            use_gcn=d.get("deform_adv_use_gcn", False),
            gcn_num_layers=d.get("deform_adv_gcn_num_layers", 2),
            num_deform_groups=d.get("deform_adv_num_deform_groups", 4),
            init_from_memory_encoder=d.get("deform_adv_init_from_memory_encoder", True),
            freeze_encoder_components=d.get("deform_adv_freeze_encoder_components", True),
            zero_mean_offsets=d.get("deform_adv_zero_mean_offsets", True),
            local_offset_gain=d.get("deform_adv_local_offset_gain", 1.3),
        )


# =============================================================================
# PGD Adversarial Configuration (Rebuttal Baseline)
# =============================================================================
@dataclass
class PGDAdvConfig:
    """Pixel-space PGD adversarial attack configuration (Madry et al.)."""

    enabled: bool = False
    epsilon: float = 0.03  # L∞ budget in normalized pixel space (~8/255)
    num_steps: int = 5  # Inner loop iterations
    step_size: float = 0.01  # Per-step magnitude (α)
    random_start: bool = True  # Random init within ε-ball

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> PGDAdvConfig:
        """Construct from flat config dict (backward compatible)."""
        return cls(
            enabled=d.get("use_pgd_adv", False),
            epsilon=d.get("pgd_adv_epsilon", 0.03),
            num_steps=d.get("pgd_adv_num_steps", 5),
            step_size=d.get("pgd_adv_step_size", 0.01),
            random_start=d.get("pgd_adv_random_start", True),
        )


# =============================================================================
# Adversarial Patch Configuration (Rebuttal Baseline)
# =============================================================================
@dataclass
class PatchAdvConfig:
    """Adversarial Patch attack (Brown et al. 2017, Nesti et al. WACV 2022)."""

    enabled: bool = False
    patch_size_ratio: float = 0.1  # Patch size as fraction of image dimension
    num_steps: int = 5  # Inner loop optimization steps
    step_size: float = 0.5  # Step size (larger than PGD since unconstrained)
    random_start: bool = True  # Random patch initialization

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> PatchAdvConfig:
        """Construct from flat config dict (backward compatible)."""
        return cls(
            enabled=d.get("use_patch_adv", False),
            patch_size_ratio=d.get("patch_adv_size_ratio", 0.1),
            num_steps=d.get("patch_adv_num_steps", 5),
            step_size=d.get("patch_adv_step_size", 0.5),
            random_start=d.get("patch_adv_random_start", True),
        )


# =============================================================================
# Random Noise Adversarial Configuration (Rebuttal Baseline)
# =============================================================================
@dataclass
class RandomNoiseAdvConfig:
    """Random noise injection baseline configuration."""

    enabled: bool = False
    epsilon: float = 0.03  # Noise magnitude in normalized pixel space
    noise_type: str = "gaussian"  # "gaussian" or "uniform"

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> RandomNoiseAdvConfig:
        """Construct from flat config dict (backward compatible)."""
        return cls(
            enabled=d.get("use_random_noise_adv", False),
            epsilon=d.get("random_noise_adv_epsilon", 0.03),
            noise_type=d.get("random_noise_adv_type", "gaussian"),
        )


# =============================================================================
# Distribution Matching Configuration
# =============================================================================
@dataclass
class MMDConfig:
    """MMD-specific parameters."""

    kernel: str = "rbf"
    bandwidth: float = 0.1
    batch_size: int = 256
    n_batches: int = 10


@dataclass
class MMDHardAwareConfig:
    """Hard-Aware MMD parameters (error-based Top-K mining)."""

    top_k_percent: float = 0.25
    max_samples: int = 4096


@dataclass
class DomainAwareSoftMMDConfig:
    """Domain-Aware Soft MMD parameters (OOD detection + MaxEnt)."""

    diversity_weight: float = 0.30
    temperature: float = 0.2
    max_samples: int = 4096
    diversity_method: Literal["channel_std", "spatial_var"] = "channel_std"
    enable_monitoring: bool = False


@dataclass
class CKAConfig:
    """CKA-specific parameters."""

    use_linear_kernel: bool = True
    use_minibatch: bool = True
    minibatch_size: int = 512


@dataclass
class GramConfig:
    """Gram matrix-specific parameters."""

    center: bool = True
    normalize: bool = True


@dataclass
class SpatialMMDConfig:
    """Spatially-Aligned MMD parameters (true local MMD with spatial correspondence)."""

    temperature: float = 0.5  # Softmax temperature for error-weighting
    kernel_bandwidth: float = 0.3  # RBF kernel bandwidth
    focus_on_hard: bool = True  # Weight patches by error magnitude
    hard_weight_power: float = 1.0  # Power for error-based weighting


@dataclass
class DistMatchingConfig:
    """Distribution matching configuration for AUE calibration loss."""

    method: Literal["mmd", "cka", "gram", "mmd_hard_aware", "domain_aware_soft_mmd", "spatial_mmd"] = "spatial_mmd"
    use_patches: bool = True
    patch_size: int = 32
    mse_weight: float = 0.3
    use_checkpoint: bool = True  # Gradient checkpointing for memory optimization

    # Method-specific configs
    mmd: MMDConfig = field(default_factory=MMDConfig)
    mmd_hard_aware: MMDHardAwareConfig = field(default_factory=MMDHardAwareConfig)
    domain_aware_soft_mmd: DomainAwareSoftMMDConfig = field(default_factory=DomainAwareSoftMMDConfig)
    cka: CKAConfig = field(default_factory=CKAConfig)
    gram: GramConfig = field(default_factory=GramConfig)
    spatial_mmd: SpatialMMDConfig = field(default_factory=SpatialMMDConfig)

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> DistMatchingConfig:
        """Construct from flat config dict (backward compatible).

        NOTE: aue_dist_matching_config is deprecated. MMD config is now
        passed directly to loss modules via scratch.mmd_config.
        """
        # Just use defaults since aue_dist_matching_config is no longer used
        dist_cfg = {}

        mmd_cfg = dist_cfg.get("mmd", {})
        hard_aware_cfg = dist_cfg.get("mmd_hard_aware", {})
        domain_aware_cfg = dist_cfg.get("domain_aware_soft_mmd", {})
        cka_cfg = dist_cfg.get("cka", {})
        gram_cfg = dist_cfg.get("gram", {})
        spatial_mmd_cfg = dist_cfg.get("spatial_mmd", {})

        # NEW: Support flat config style (temperature, kernel_bandwidth at top level)
        # This is the current YAML format
        if "temperature" in dist_cfg and "spatial_mmd" not in dist_cfg:
            spatial_mmd_cfg = {
                "temperature": dist_cfg.get("temperature", 0.5),
                "kernel_bandwidth": dist_cfg.get("kernel_bandwidth", 0.3),
                "focus_on_hard": dist_cfg.get("focus_on_hard", True),
                "hard_weight_power": dist_cfg.get("hard_weight_power", 1.0),
            }

        return cls(
            method=dist_cfg.get("method", "spatial_mmd"),
            use_patches=dist_cfg.get("use_patches", True),
            patch_size=dist_cfg.get("patch_size", 32),
            mse_weight=dist_cfg.get("mse_weight", 0.3),
            use_checkpoint=dist_cfg.get("use_checkpoint", True),
            mmd=MMDConfig(
                kernel=mmd_cfg.get("kernel", "rbf"),
                bandwidth=mmd_cfg.get("bandwidth", 0.1),
                batch_size=mmd_cfg.get("batch_size", 256),
                n_batches=mmd_cfg.get("n_batches", 10),
            ),
            mmd_hard_aware=MMDHardAwareConfig(
                top_k_percent=hard_aware_cfg.get("top_k_percent", 0.25),
                max_samples=hard_aware_cfg.get("max_samples", 4096),
            ),
            domain_aware_soft_mmd=DomainAwareSoftMMDConfig(
                diversity_weight=domain_aware_cfg.get("diversity_weight", 0.30),
                temperature=domain_aware_cfg.get("temperature", 0.2),
                max_samples=domain_aware_cfg.get("max_samples", 4096),
                diversity_method=domain_aware_cfg.get("diversity_method", "channel_std"),
                enable_monitoring=domain_aware_cfg.get("enable_monitoring", False),
            ),
            cka=CKAConfig(
                use_linear_kernel=cka_cfg.get("use_linear_kernel", True),
                use_minibatch=cka_cfg.get("use_minibatch", True),
                minibatch_size=cka_cfg.get("minibatch_size", 512),
            ),
            gram=GramConfig(
                center=gram_cfg.get("center", True),
                normalize=gram_cfg.get("normalize", True),
            ),
            spatial_mmd=SpatialMMDConfig(
                temperature=spatial_mmd_cfg.get("temperature", 0.5),
                kernel_bandwidth=spatial_mmd_cfg.get("kernel_bandwidth", 0.3),
                focus_on_hard=spatial_mmd_cfg.get("focus_on_hard", True),
                hard_weight_power=spatial_mmd_cfg.get("hard_weight_power", 1.0),
            ),
        )

    def get_max_samples(self) -> int:
        """Get max_samples based on active method."""
        if self.method == "domain_aware_soft_mmd":
            return self.domain_aware_soft_mmd.max_samples
        return self.mmd_hard_aware.max_samples


# =============================================================================
# Master AUE Configuration
# =============================================================================
@dataclass
class AUEConfig:
    """
    Master AUE configuration, aggregates all sub-configs.

    This is the single entry point for all AUE-related configuration.
    It replaces 60+ individual parameters in SAM2Base.__init__.

    Usage:
        # Backward compatible: from existing model kwargs
        aue_cfg = AUEConfig.from_model_kwargs(**model_kwargs)

        # Direct construction with nested configs
        aue_cfg = AUEConfig(
            enabled=True,
            style=StyleAdvConfig(enabled=True, epsilon=1.5),
        )
    """

    # Master enable flags
    enabled: bool = False  # use_aue
    # Alternating training: probability of running AUE branch per step (0~1)
    probability: float = 0.5  # aue_probability
    use_analytic_uncertainty: bool = True

    # Multi-object control
    use_multi_object: bool = True
    enable_background: bool = True

    # Legacy AUE options (kept for backward compatibility)
    num_adversarial_samples: int = 32
    init_from_dataset: bool = False
    diversity_loss_weight: float = 0.0

    # Max objects (matches dataset sampler)
    max_num_objects: int = 11  # 10 objects + 1 background

    # Sub-configs
    style: StyleAdvConfig = field(default_factory=StyleAdvConfig)
    deform: DeformAdvConfig = field(default_factory=DeformAdvConfig)
    pgd: PGDAdvConfig = field(default_factory=PGDAdvConfig)
    patch: PatchAdvConfig = field(default_factory=PatchAdvConfig)
    random_noise: RandomNoiseAdvConfig = field(default_factory=RandomNoiseAdvConfig)
    dist_matching: DistMatchingConfig = field(default_factory=DistMatchingConfig)

    # Attack ordering
    attack_order: list[str] = field(default_factory=lambda: ["style", "deform"])

    @classmethod
    def from_model_kwargs(cls, **kwargs: Any) -> AUEConfig:
        """
        Construct AUEConfig from SAM2Base.__init__ kwargs.

        This provides backward compatibility with existing YAML configs
        that pass flat parameters to the model constructor.

        Args:
            **kwargs: Model constructor arguments (from Hydra config)

        Returns:
            Fully constructed AUEConfig
        """
        return cls(
            enabled=kwargs.get("use_aue", False),
            probability=kwargs.get("aue_probability", 0.5),
            use_analytic_uncertainty=kwargs.get("aue_use_analytic_uncertainty", True),
            use_multi_object=kwargs.get("adv_use_multi_object", True),
            enable_background=kwargs.get("adv_enable_background", True),
            num_adversarial_samples=kwargs.get("aue_num_adversarial_samples", 32),
            init_from_dataset=kwargs.get("aue_init_from_dataset", False),
            diversity_loss_weight=kwargs.get("aue_diversity_loss_weight", 0.0),
            max_num_objects=kwargs.get("max_num_objects", 11),
            style=StyleAdvConfig.from_flat_dict(kwargs),
            deform=DeformAdvConfig.from_flat_dict(kwargs),
            pgd=PGDAdvConfig.from_flat_dict(kwargs),
            patch=PatchAdvConfig.from_flat_dict(kwargs),
            random_noise=RandomNoiseAdvConfig.from_flat_dict(kwargs),
            dist_matching=DistMatchingConfig.from_flat_dict(kwargs),
            attack_order=kwargs.get("adversarial_attack_order", ["style", "deform"]),
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Convert back to flat dict format (for logging/debugging).

        Returns:
            Flat dict with original parameter names
        """
        return {
            "use_aue": self.enabled,
            "aue_use_analytic_uncertainty": self.use_analytic_uncertainty,
            "adv_use_multi_object": self.use_multi_object,
            "adv_enable_background": self.enable_background,
            "aue_num_adversarial_samples": self.num_adversarial_samples,
            "aue_init_from_dataset": self.init_from_dataset,
            "aue_diversity_loss_weight": self.diversity_loss_weight,
            "max_num_objects": self.max_num_objects,
            # Style
            "use_style_adv": self.style.enabled,
            "style_adv_mode": self.style.mode,
            "style_adv_epsilon": self.style.epsilon,
            "style_adv_use_gt_region_style": self.style.use_gt_region_style,
            "style_adv_use_global_local_mix": self.style.use_global_local_mix,
            "style_adv_global_epsilon": self.style.global_epsilon,
            "style_adv_global_weight": self.style.global_weight,
            "style_adv_use_gcn": self.style.gcn.enabled,
            # Deform
            "use_deform_adv": self.deform.enabled,
            "deform_adv_epsilon": self.deform.epsilon,
            # PGD
            "use_pgd_adv": self.pgd.enabled,
            "pgd_adv_epsilon": self.pgd.epsilon,
            "pgd_adv_num_steps": self.pgd.num_steps,
            # Random Noise
            "use_random_noise_adv": self.random_noise.enabled,
            "random_noise_adv_epsilon": self.random_noise.epsilon,
            "adversarial_attack_order": self.attack_order,
        }

    @property
    def any_adversarial_enabled(self) -> bool:
        """Check if any adversarial attack is enabled."""
        return self.style.enabled or self.deform.enabled or self.pgd.enabled or self.random_noise.enabled
