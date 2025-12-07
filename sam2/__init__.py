# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from hydra import initialize_config_module, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    # Check if we should use experiment-specific configs instead of default pkg://sam2
    experiment_config_dir = os.environ.get("EXPERIMENT_CONFIG_DIR")
    
    if experiment_config_dir and os.path.exists(experiment_config_dir):
        # Use experiment directory as the config source
        # This ensures we load the correct configs from the start
        initialize_config_dir(config_dir=experiment_config_dir, version_base="1.2")
        print(f"[sam2.__init__] Initialized Hydra with experiment config directory: {experiment_config_dir}")
    else:
        # Use default pkg://sam2 config module
        initialize_config_module("sam2", version_base="1.2")
