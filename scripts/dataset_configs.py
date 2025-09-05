#!/usr/bin/env python3
"""
Shared dataset configurations for SAM-2 zero-shot evaluation scripts.
This ensures consistency between different evaluation approaches.
"""

# Dataset configurations for zero-shot evaluation
DATASET_CONFIGS = {
    "TrashCan": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/TrashCan_SAM2/",
        "splits": ["train", "val"],
        "default_split": "val",
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "GTEA": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/GTEA_SAM2/",
        "splits": ["train", "val"],
        "default_split": "val",
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "PIDRay": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/PIDRay_SAM2/",
        "splits": ["train", "test"],
        "default_split": "test",
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "plittersdorf": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/plittersdorf_SAM2/",
        "splits": ["train", "val", "test"],
        "default_split": "test",
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "Hypersim": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/Hypersim_SAM2/",
        "splits": ["train"],
        "default_split": "train",
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "DRAM": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/DRAM_SAM2/",
        "splits": ["train", "test"],
        "default_split": "test",
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "CITYSCAPES": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/CITYSCAPES_SAM2/",
        "splits": ["train", "val", "test"],
        "default_split": "val",  # Use val split like SAM paper (dense annotations)
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "IBD": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/IBD_SAM2/",
        "splits": ["train", "val"],
        "default_split": "val",
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
    "NDISPark": {
        "root": "/ssdArray/hongyou/dev/data/sam2_data/NDISPark_SAM2/",
        "splits": ["train", "validation"],
        "default_split": "validation",
        "has_split_subdir": True,
        "skip_first_and_last": False,
    },
}

# Default dataset list for evaluation
DEFAULT_DATASETS = [
    "TrashCan",
    "GTEA",
    "PIDRay",
    "plittersdorf",
    "Hypersim",
    "DRAM",
    "CITYSCAPES",
    "IBD",
    "NDISPark",
]
