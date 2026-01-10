#!/usr/bin/env python3
"""
Shared dataset configurations for SAM-2 zero-shot evaluation scripts.
This ensures consistency between different evaluation approaches.
"""

from __future__ import annotations

# Dataset configurations for zero-shot evaluation
DATASET_CONFIGS = {
    "TrashCan": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/TrashCan_SAM2/",
        "splits": ["train", "val"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "GTEA": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/GTEA_SAM2/",
        "splits": ["train", "val"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "PIDRay": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/PIDRay_SAM2/",
        "splits": ["train", "test"],
        "default_split": ["test"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "plittersdorf": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/plittersdorf_SAM2/",
        "splits": ["train", "val", "test"],
        "default_split": ["train", "val", "test"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "Hypersim": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/Hypersim_SAM2/",
        "splits": ["train"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "DRAM": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/DRAM_SAM2/",
        "splits": ["train", "test", "val"],
        "default_split": ["test"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "CITYSCAPES": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/CITYSCAPES_SAM2/",
        "splits": ["train", "val", "test"],
        "default_split": ["val"],  # Use val split like SAM paper (dense annotations)
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "IBD": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/IBD_SAM2/",
        "splits": ["train", "val"],
        "default_split": ["val"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "NDISPark": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/NDISPark_SAM2/",
        "splits": ["train", "validation"],
        "default_split": ["train", "validation"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "WoodScape": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/WoodScape_SAM2/",
        "splits": ["train", "val"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "EgoHOS": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/EgoHOS_SAM2/",
        "splits": ["train", "val", "test_indomain", "test_outdomain"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "VISOR": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/VISOR_SAM2/",
        "splits": ["train", "val", "test"],
        "default_split": ["val"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "OVIS": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/OVIS_SAM2/",
        "splits": ["train"],
        "default_split": ["train"],
        "has_split_subdir": True,  # OVIS has train/Annotations/JPEGImages structure
        "skip_first_and_last": False,
    },
    "ADE20K": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/ADE20K_SAM2/",
        "splits": ["training", "validation"],
        "default_split": ["validation"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "iShape": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/iShape_SAM2/",
        "splits": ["train", "val"],
        "default_split": ["val"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "ZeroWaste-f": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/ZeroWaste-f_SAM2/",
        "splits": ["train", "val", "test"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    # New datasets
    "LVIS": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/LVIS_SAM2/",
        "splits": ["val"],
        "default_split": ["val"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "NDD20": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/NDD20_SAM2/",
        "splits": ["train"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "TimberSeg": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/TimberSeg_SAM2/",
        "splits": ["train"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "DOORS": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/DOORS_SAM2/",
        "splits": ["train", "val", "test1", "test2"],
        "default_split": ["val"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "BBBC038v1": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/BBBC038v1_SAM2/",
        "splits": ["train"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "PPDLS": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/PPDLS_SAM2/",
        "splits": ["train"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "STREETS": {
        "root": "/home/hongyou/dev/data/sam2_data/STREETS_SAM2/",
        "splits": ["train"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    # MOSE split for domain shift analysis
    # Both use MOSE official train set (val set has no GT), split via different file lists
    "MOSE_train": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/MOSE_release/",
        "splits": ["train"],
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
        "file_list_txt": "/home/hongyou/dev/ada_samp/sam2/training/assets/MOSE_sample_train_list.txt",  # 1246 videos (86%)
        "note": "Source domain for BNDL/UR-ERN fine-tuning",
    },
    "MOSE_val": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/MOSE_release/",
        "splits": ["train"],  # Note: still uses 'train' directory (official val has no GT)
        "default_split": ["train"],
        "has_split_subdir": True,
        "skip_first_and_last": False,
        "file_list_txt": "/home/hongyou/dev/ada_samp/sam2/training/assets/MOSE_sample_val_list.txt",  # 200 videos (14%)
        "note": "Within-domain baseline for domain shift analysis",
    },
}

# Default dataset list for evaluation
DEFAULT_DATASETS = [
    # "MOSE_train",  # Source domain (fine-tune domain for BNDL/UR-ERN, 1246 videos)
    # "MOSE_val",    # Within-domain baseline (200 videos)
    "TrashCan",
    "GTEA",
    "PIDRay",
    "plittersdorf",
    "Hypersim",
    "DRAM",
    "CITYSCAPES",
    "IBD",
    "NDISPark",
    "WoodScape",
    "EgoHOS",
    "VISOR",
    "OVIS",
    "ADE20K",
    "iShape",
    "ZeroWaste-f",
    "LVIS",
    "NDD20",
    "TimberSeg",
    "DOORS",
    "BBBC038v1",
    "PPDLS",
    "STREETS",
]

# -----------------------------------------------------------------------------
# Centralized dataset type taxonomy (3 types)
# -----------------------------------------------------------------------------
# Category keys (stable identifiers) mapped to human-readable labels
DATASET_TYPE_CATEGORIES: dict[str, str] = {
    "OBJECT": "Object-centric / Instance",
    "SCENE": "Scene-centric / Stuff",
    "EGO_VIDEO": "Egocentric / Video",
}

# Mapping from dataset name (as used in DATASET_CONFIGS) to a category key above
DATASET_TO_TYPE: dict[str, str] = {
    # Object-centric / Instance
    "TrashCan": "OBJECT",
    "PIDRay": "OBJECT",
    "iShape": "OBJECT",
    "ZeroWaste-f": "OBJECT",
    "LVIS": "OBJECT",
    "DOORS": "OBJECT",
    "BBBC038v1": "OBJECT",
    "PPDLS": "OBJECT",
    "IBD": "OBJECT",

    # Scene-centric / Stuff
    "CITYSCAPES": "SCENE",
    "ADE20K": "SCENE",
    "Hypersim": "SCENE",
    "WoodScape": "SCENE",
    "STREETS": "SCENE",
    "NDISPark": "SCENE",
    "plittersdorf": "SCENE",
    "NDD20": "SCENE",
    "TimberSeg": "SCENE",

    # Egocentric / Video
    "GTEA": "EGO_VIDEO",
    "EgoHOS": "EGO_VIDEO",
    "VISOR": "EGO_VIDEO",
    "OVIS": "EGO_VIDEO",
    "MOSE_train": "EGO_VIDEO",  # MOSE train split (1246 videos) - fine-tune domain
    "MOSE_val": "EGO_VIDEO",    # MOSE val split (200 videos) - within-domain test
    "DRAM": "EGO_VIDEO",
}
