# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AUE (Adversarial Uncertainty Estimation) Module.

Provides structured configuration and utilities for adversarial training
in SAM2 with uncertainty estimation.

Main exports:
    - AUEModule: Main orchestrator using composition pattern
    - AUEConfig: Master configuration dataclass
    - StyleAdvConfig: Style adversarial attack configuration
    - DeformAdvConfig: Deformation adversarial attack configuration
    - DistMatchingConfig: Distribution matching configuration
    - AUELossComputer: Loss computation utilities
    - AdversarialPipeline: Attack pipeline orchestrator
    - AUEVisualizer: Visualization utilities
"""

from sam2.modeling.aue.config import (
    AUEConfig,
    CKAConfig,
    DeformAdvConfig,
    DistMatchingConfig,
    DomainAwareSoftMMDConfig,
    GramConfig,
    MMDConfig,
    MMDHardAwareConfig,
    StyleAdvConfig,
    StyleGCNConfig,
)
from sam2.modeling.aue.loss import (
    AUELossComputer,
    extract_bndl_outputs,
    prepare_gt_for_loss,
)
from sam2.modeling.aue.module import AUEModule
from sam2.modeling.aue.pipeline import AdversarialPipeline, cleanup_augmentation_results
from sam2.modeling.aue.visualization import AUEVisualizationData, AUEVisualizer

__all__ = [
    # Main module
    "AUEModule",
    # Master config
    "AUEConfig",
    # Adversarial attack configs
    "StyleAdvConfig",
    "StyleGCNConfig",
    "DeformAdvConfig",
    # Distribution matching configs
    "DistMatchingConfig",
    "MMDConfig",
    "MMDHardAwareConfig",
    "DomainAwareSoftMMDConfig",
    "CKAConfig",
    "GramConfig",
    # Loss computation
    "AUELossComputer",
    "extract_bndl_outputs",
    "prepare_gt_for_loss",
    # Pipeline
    "AdversarialPipeline",
    "cleanup_augmentation_results",
    # Visualization
    "AUEVisualizationData",
    "AUEVisualizer",
]
