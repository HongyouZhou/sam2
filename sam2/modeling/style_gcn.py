# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GCN module for refining multi-object style perturbations.

The GCN coordinates perturbations between spatially or semantically related objects,
making adversarial attacks more natural while preserving effectiveness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdversarialStyleGCN(nn.Module):
    """
    Graph Convolutional Network for refining style perturbations across multiple objects.

    Uses a 2-layer GCN with residual connections to coordinate perturbations between
    spatially or semantically related objects. Optionally accepts visual features
    to enable content-aware style refinement.

    Args:
        style_dim: Dimension of style perturbation vector (6 for Gram matrix style)
        feature_dim: Dimension of visual features (0 = no features, 256 for backbone_fpn[-1])
        num_layers: Number of GCN layers (default: 2)
    """

    def __init__(self, style_dim: int = 6, feature_dim: int = 0, num_layers: int = 2):
        super().__init__()

        self.style_dim = style_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # Node dimension: style + optional visual features
        self.node_dim = style_dim + feature_dim

        # Build GCN layers
        # First num_layers-1 layers maintain node_dim for processing
        # Last layer projects back to style_dim only
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                # Intermediate layers: maintain node dimension
                layer = nn.Linear(self.node_dim, self.node_dim)
            else:
                # Last layer: project back to style space
                layer = nn.Linear(self.node_dim, style_dim)

            # Initialize all layers with small weights for near-identity behavior
            # Combined with alpha ≈ 0, this makes the GCN output close to input initially
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # Small random init
            nn.init.zeros_(layer.bias)

            self.layers.append(layer)

        # Layer normalization for stability
        # Use node_dim for intermediate layers, style_dim for last layer
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.layer_norms.append(nn.LayerNorm(self.node_dim))
            else:
                self.layer_norms.append(nn.LayerNorm(style_dim))

        # Learnable residual weights (alpha) per layer
        # Initialize to -2.0 for moderate near-identity mapping at start
        # sigmoid(-2) ≈ 0.12, so output ≈ 0.12 * GCN(x) + 0.88 * x
        # This allows GCN to have meaningful impact from the start while
        # still being conservative enough not to disrupt PGD too much
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor([-2.0])) for _ in range(num_layers)])

    def forward(self, style_deltas: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                mask_features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of GCN with optional visual features.

        Args:
            style_deltas: Style perturbations [B, num_nodes, style_dim]
                         where num_nodes = K (foreground only) or K+1 (with background)
            edge_index: Edge indices [2, E] where E is number of edges
            edge_weight: Edge weights [E]
            mask_features: Optional visual features [B, num_nodes, feature_dim]
                          extracted from image encoder for each mask region

        Returns:
            Refined style perturbations [B, num_nodes, style_dim]
        """
        B, num_nodes, _ = style_deltas.shape

        # Concatenate visual features if provided
        if mask_features is not None:
            assert mask_features.shape[:2] == (B, num_nodes), \
                f"mask_features shape {mask_features.shape} doesn't match style_deltas batch/nodes ({B}, {num_nodes})"
            assert mask_features.shape[2] == self.feature_dim, \
                f"mask_features dim {mask_features.shape[2]} doesn't match feature_dim {self.feature_dim}"
            node_features = torch.cat([style_deltas, mask_features], dim=-1)  # [B, num_nodes, node_dim]
        else:
            node_features = style_deltas  # [B, num_nodes, style_dim]

        # Flatten batch and nodes for processing
        x = node_features.reshape(B * num_nodes, self.node_dim)  # [B*num_nodes, node_dim]

        # Apply GCN layers
        # Intermediate layers maintain node_dim, last layer projects to style_dim
        for layer_idx, (linear, norm, alpha) in enumerate(zip(self.layers, self.layer_norms, self.alphas)):
            # x_input = x  # Save input for residual connection

            # Message passing (use actual num_nodes from input shape)
            x = self._graph_conv(x, edge_index, edge_weight, B, num_nodes)

            # Linear transformation
            x = linear(x)

            # ReLU activation (except last layer)
            if layer_idx < self.num_layers - 1:
                # Layer normalization
                x = norm(x) 
                x = F.relu(x)

            # # Residual connection with learnable weight (works for all layers now)
            # alpha_val = torch.sigmoid(alpha)  # Ensure alpha in [0, 1]
            # x = alpha_val * x + (1 - alpha_val) * x_input  ### TODO: problem unbounded output value

        # Reshape back to [B, num_nodes, style_dim]
        # Last layer has projected back to style_dim, so output is style perturbations only
        x = x.reshape(B, num_nodes, self.style_dim)

        return x

    def _graph_conv(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch_size: int, num_nodes_per_batch: int) -> torch.Tensor:
        """
        Perform standard graph convolution operation.

        This method performs a uniform graph convolution on all nodes,
        without any special handling based on node semantics (foreground/background).
        The node features should already include all nodes that participate in the graph.

        Args:
            x: Node features [num_nodes, D] where num_nodes = B * num_nodes_per_batch
            edge_index: Edge indices [2, E] where E is number of edges
            edge_weight: Edge weights [E]
            batch_size: Batch size B
            num_nodes_per_batch: Number of nodes per batch (K or K+1 depending on config)

        Returns:
            Aggregated features [num_nodes, D]
        """
        if edge_index.numel() == 0:
            # No edges, return identity
            return x

        # Extract source and target nodes
        src, tgt = edge_index[0], edge_index[1]

        # Allocate output tensor (use new_zeros for memory efficiency)
        out = x.new_zeros(x.size(0), x.size(1))

        # Gather source features and apply edge weights
        src_features = x[src]  # [E, D]
        weighted_features = src_features * edge_weight.unsqueeze(1)  # [E, D]

        # Aggregate messages to target nodes
        out.index_add_(0, tgt, weighted_features)  # out[tgt] += weighted_features

        return out


def compute_boundary_distance(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    grid_y: torch.Tensor,
    grid_x: torch.Tensor,
) -> float:
    """
    Compute minimum distance between boundaries of two masks.

    Args:
        mask1: Binary mask [H, W]
        mask2: Binary mask [H, W]
        grid_y: Y coordinate grid [1, 1, H, W]
        grid_x: X coordinate grid [1, 1, H, W]

    Returns:
        Minimum distance between boundaries (normalized by image diagonal)
    """
    # Extract boundary pixels using dilation
    # boundary = dilated_mask - mask
    kernel_size = 3
    padding = kernel_size // 2

    # Dilate masks to find boundaries
    mask1_expanded = mask1.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask2_expanded = mask2.unsqueeze(0).unsqueeze(0)

    # Use max_pool2d as dilation operation with padding to maintain size
    # padding=kernel_size//2 ensures output size matches input size
    mask1_dilated = F.max_pool2d(mask1_expanded, kernel_size=kernel_size, stride=1, padding=padding)
    mask2_dilated = F.max_pool2d(mask2_expanded, kernel_size=kernel_size, stride=1, padding=padding)

    # Boundary = dilated - original (both should be [1, 1, H, W] now)
    boundary1 = (mask1_dilated - mask1_expanded).squeeze() > 0.5  # [H, W] bool
    boundary2 = (mask2_dilated - mask2_expanded).squeeze() > 0.5  # [H, W] bool

    # If no boundary pixels, fall back to centroid distance
    if not boundary1.any() or not boundary2.any():
        # Compute centroids as fallback
        area1 = mask1.sum()
        area2 = mask2.sum()
        if area1 <= 0 or area2 <= 0:
            return float("inf")

        cy1 = (mask1 * grid_y[0, 0]).sum() / area1
        cx1 = (mask1 * grid_x[0, 0]).sum() / area1
        cy2 = (mask2 * grid_y[0, 0]).sum() / area2
        cx2 = (mask2 * grid_x[0, 0]).sum() / area2

        dist = ((cy1 - cy2) ** 2 + (cx1 - cx2) ** 2) ** 0.5
        return float(dist)

    # Get boundary pixel coordinates
    boundary1_coords = torch.stack([grid_y[0, 0][boundary1], grid_x[0, 0][boundary1]], dim=1)  # [N1, 2]
    boundary2_coords = torch.stack([grid_y[0, 0][boundary2], grid_x[0, 0][boundary2]], dim=1)  # [N2, 2]

    if boundary1_coords.shape[0] == 0 or boundary2_coords.shape[0] == 0:
        return float("inf")

    # Compute pairwise distances between boundary pixels
    # Expand dimensions for broadcasting: [N1, 1, 2] - [1, N2, 2] = [N1, N2, 2]
    diff = boundary1_coords.unsqueeze(1) - boundary2_coords.unsqueeze(0)  # [N1, N2, 2]
    distances = (diff**2).sum(dim=2) ** 0.5  # [N1, N2]

    # Return minimum distance
    min_dist = distances.min().item()
    return min_dist


def build_object_graph(
    masks: torch.Tensor,
    img_batch: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    edge_threshold: float = 0.3,
    use_semantic: bool = True,
    use_background: bool = False,
    distance_threshold: float | None = None,
    use_boundary_distance: bool = False,  # Use boundary distance instead of centroid distance
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Build object graph from masks for GCN message passing.

    This version respects the real object count per sample (padding slots are ignored),
    treats the background as the dedicated final slot when `use_background=True`, and
    optionally connects foreground nodes based on distance (centroid or boundary).

    Args:
        masks: Binary masks [B, K, H, W] or [B, K, 1, H, W]
        img_batch: Optional images (unused, reserved for future semantic cues)
        labels: Optional object labels [B, K] for semantic edges (currently unused)
        edge_threshold: IoU threshold for spatial edges
        use_semantic: Whether to add semantic edges between same-label objects
        use_background: Whether to connect foreground objects with the background node
        distance_threshold: Distance threshold for geometric adjacency (normalized to [0,1])
        use_boundary_distance: If True, use boundary distance; if False, use centroid distance

    Returns:
        edge_index: Edge indices [2, E] in COO format
        edge_weight: Edge weights [E], normalized by degree
        stats: Dictionary of summary statistics for logging/debugging
    """
    if masks.ndim == 5:
        masks = masks.squeeze(2)

    B, K, H, W = masks.shape
    device = masks.device

    masks_binary = (masks > 0.5).float()
    mask_areas = masks_binary.sum(dim=(2, 3))  # [B, K]

    # Pre-compute coordinate grids for distance calculation (centroid or boundary)
    if (distance_threshold is not None and distance_threshold > 0) or use_boundary_distance:
        y_coords = torch.arange(H, device=device, dtype=masks_binary.dtype)
        x_coords = torch.arange(W, device=device, dtype=masks_binary.dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)
        diag = float(torch.sqrt(torch.tensor(float(H * H + W * W)))) if (H > 0 and W > 0) else 1.0
        # Clamp to avoid division by zero
        diag = max(diag, 1.0)
    else:
        grid_y = grid_x = None
        diag = 1.0

    nodes_per_batch = K
    has_background_slot = use_background and K > 0
    background_slot = nodes_per_batch - 1 if has_background_slot else None

    edge_src: list[int] = []
    edge_tgt: list[int] = []
    edge_wgt: list[float] = []

    stats = {
        "graphs": 0,
        "nodes_total": 0,
        "nodes_foreground": 0,
        "nodes_background": 0,
        "edges_total": 0,
        "edges_iou": 0,
        "edges_distance": 0,
        "edges_background": 0,
    }

    for b in range(B):
        # Foreground slots: everything before the dedicated background slot (if any)
        if has_background_slot:
            fg_range = range(background_slot)
            background_area = mask_areas[b, background_slot]
        else:
            fg_range = range(nodes_per_batch)
            background_area = torch.tensor(0.0, device=device)

        object_indices = [idx for idx in fg_range if mask_areas[b, idx] > 0]
        background_idx = background_slot if has_background_slot and background_area > 0 else None
        
        # Debug: log object_indices and background_idx
        if not object_indices and background_idx is None:
            # Log why this batch was skipped (only for first batch to avoid spam)
            if b == 0:
                import logging

                total_area = mask_areas[b].sum().item()
                # Check raw mask values to understand the issue
                raw_mask_sum = masks[b].sum().item()
                max_val = masks[b].max().item()
                min_val = masks[b].min().item()
                above_threshold = (masks[b] > 0.5).sum().item()
                # Show per-channel area distribution
                area_per_channel = [f"{i}:{mask_areas[b, i].item():.0f}" for i in range(min(K, 11))]
                logging.warning(
                    f"GCN: Skipping batch {b} - no valid nodes. "
                    f"has_bg_slot={has_background_slot}, bg_slot={background_slot if has_background_slot else 'N/A'}, "
                    f"bg_area={background_area.item() if isinstance(background_area, torch.Tensor) else 0:.0f}, "
                    f"fg_objects=0/{K}, areas=[{', '.join(area_per_channel)}], "
                    f"raw_sum={raw_mask_sum:.0f}, range=[{min_val:.3f}, {max_val:.3f}]"
                )
            continue

        # Debug log for first batch only
        if b == 0:
            import logging

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                # Show detailed mask area info
                mask_area_list = [f"{i}:{mask_areas[b, i].item():.0f}" for i in range(K)]
                logging.debug(f"GCN batch {b}: mask_areas=[{', '.join(mask_area_list)}]")
                logging.debug(
                    f"GCN batch {b}: object_indices={object_indices}, background_idx={background_idx}, "
                    f"background_slot={background_slot if has_background_slot else None}, "
                    f"background_area={background_area.item() if background_area is not None else 0:.0f}"
                )

        stats["graphs"] += 1
        stats["nodes_foreground"] += len(object_indices)
        stats["nodes_background"] += int(background_idx is not None)
        stats["nodes_total"] += len(object_indices) + int(background_idx is not None)

        # Spatial edges between foreground objects (IoU-based)
        for i_pos, i in enumerate(object_indices):
            mask_i = masks_binary[b, i]
            area_i = mask_areas[b, i]
            if area_i <= 0:
                continue

            for j in object_indices[i_pos + 1 :]:
                mask_j = masks_binary[b, j]
                area_j = mask_areas[b, j]
                if area_j <= 0:
                    continue

                intersection = (mask_i * mask_j).sum()
                if intersection <= 0:
                    continue

                union = area_i + area_j - intersection
                if union <= 0:
                    continue

                iou = (intersection / union).item()
                if iou <= edge_threshold:
                    continue

                # Debug log for first IoU edge found in first batch
                if b == 0 and stats["edges_iou"] == 0:
                    import logging

                    logging.debug(f"GCN: First IoU edge found: obj_{i} <-> obj_{j}, iou={iou:.3f}, threshold={edge_threshold}")

                node_i = b * nodes_per_batch + i
                node_j = b * nodes_per_batch + j
                edge_src.extend([node_i, node_j])
                edge_tgt.extend([node_j, node_i])
                edge_wgt.extend([iou, iou])
                stats["edges_iou"] += 2

        # Geometric adjacency based on distance (centroid or boundary)
        if distance_threshold is not None and distance_threshold > 0 and len(object_indices) > 1:
            if use_boundary_distance:
                # Use boundary distance
                fg_list = object_indices
                for i_pos, i in enumerate(fg_list):
                    mask_i = masks_binary[b, i]
                    area_i = mask_areas[b, i]
                    if area_i <= 0:
                        continue

                    node_i = b * nodes_per_batch + i
                    for j in fg_list[i_pos + 1 :]:
                        mask_j = masks_binary[b, j]
                        area_j = mask_areas[b, j]
                        if area_j <= 0:
                            continue

                        # Compute boundary distance
                        boundary_dist = compute_boundary_distance(mask_i, mask_j, grid_y, grid_x)
                        norm_dist = boundary_dist / diag

                        if norm_dist > distance_threshold:
                            continue

                        # Debug log for first boundary distance edge found in first batch
                        if b == 0 and stats["edges_distance"] == 0:
                            import logging

                            logging.debug(f"GCN: First boundary distance edge found: obj_{i} <-> obj_{j}, norm_dist={norm_dist:.3f}, threshold={distance_threshold}")

                        # Weight decreases linearly with normalized distance, minimum epsilon
                        weight = max(1e-6, (distance_threshold - norm_dist) / distance_threshold)
                        node_j = b * nodes_per_batch + j
                        edge_src.extend([node_i, node_j])
                        edge_tgt.extend([node_j, node_i])
                        edge_wgt.extend([weight, weight])
                        stats["edges_distance"] += 2
            else:
                # Use centroid distance (original implementation)
                centroids: dict[int, tuple[float, float]] = {}
                for idx in object_indices:
                    area = mask_areas[b, idx]
                    if area <= 0:
                        continue
                    mask = masks_binary[b, idx]
                    cy = float((mask * grid_y[0, 0]).sum() / area)
                    cx = float((mask * grid_x[0, 0]).sum() / area)
                    centroids[idx] = (cy, cx)

                fg_list = list(centroids.keys())
                for i_pos, i in enumerate(fg_list):
                    ci = centroids[i]
                    node_i = b * nodes_per_batch + i
                    for j in fg_list[i_pos + 1 :]:
                        cj = centroids[j]
                        node_j = b * nodes_per_batch + j
                        dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
                        norm_dist = dist / diag
                        if norm_dist > distance_threshold:
                            continue

                        # Debug log for first distance edge found in first batch
                        if b == 0 and stats["edges_distance"] == 0:
                            import logging

                            logging.debug(f"GCN: First centroid distance edge found: obj_{i} <-> obj_{j}, norm_dist={norm_dist:.3f}, threshold={distance_threshold}")

                        # Weight decreases linearly with normalized distance, minimum epsilon
                        weight = max(1e-6, (distance_threshold - norm_dist) / distance_threshold)
                        edge_src.extend([node_i, node_j])
                        edge_tgt.extend([node_j, node_i])
                        edge_wgt.extend([weight, weight])
                        stats["edges_distance"] += 2

        # Optional background edges (foreground ↔ background)
        if background_idx is not None and object_indices:
            node_bg = b * nodes_per_batch + background_idx
            for obj_idx in object_indices:
                node_obj = b * nodes_per_batch + obj_idx
                edge_src.extend([node_obj, node_bg])
                edge_tgt.extend([node_bg, node_obj])
                edge_wgt.extend([0.5, 0.5])
                stats["edges_background"] += 2

            # Debug log for first batch
            if b == 0:
                import logging

                logging.debug(f"GCN: Added {len(object_indices) * 2} background edges connecting {len(object_indices)} objects to background node {background_idx}")
        elif b == 0:
            import logging

            logging.debug(f"GCN: No background edges - background_idx={background_idx}, object_indices={object_indices}")

    if len(edge_src) == 0:
        # Debug: log why no edges were created despite having stats
        if stats["graphs"] > 0 or stats["nodes_total"] > 0:
            import logging

            # Only warn if there are foreground nodes; pure background is expected for non-cond frames
            if stats["nodes_foreground"] > 0:
                logging.warning(
                    f"GCN: Had {stats['graphs']:.0f} graphs with {stats['nodes_total']:.0f} nodes ({stats['nodes_foreground']:.0f}fg+{stats['nodes_background']:.0f}bg) but NO edges were created!"
                )
            else:
                logging.debug(f"GCN: Only background nodes ({stats['nodes_background']:.0f}bg), no edges created (expected).")
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            {
                "graphs": 0.0,
                "nodes_total": 0.0,
                "nodes_foreground": 0.0,
                "nodes_background": 0.0,
                "edges_total": 0.0,
                "edges_iou": 0.0,
                "edges_distance": 0.0,
                "edges_background": 0.0,
                "avg_degree": 0.0,
                "avg_fg_degree": 0.0,
            },
        )

    edge_src_t = torch.tensor(edge_src, dtype=torch.long, device=device)
    edge_tgt_t = torch.tensor(edge_tgt, dtype=torch.long, device=device)
    edge_wgt_t = torch.tensor(edge_wgt, dtype=torch.float32, device=device)

    edge_index = torch.stack([edge_src_t, edge_tgt_t], dim=0)

    num_nodes = B * K
    degree = torch.zeros(num_nodes, device=device, dtype=edge_wgt_t.dtype)
    degree.index_add_(0, edge_tgt_t, edge_wgt_t)
    # compute degree stats before clamping for normalization
    if stats["graphs"] > 0:
        degree_cpu = degree.detach().cpu()
        active = degree_cpu > 0
        total_edges = float(len(edge_src_t))
        stats["edges_total"] = total_edges
        stats["avg_degree"] = float(degree_cpu.mean().item())
        if active.any():
            stats["avg_fg_degree"] = float(degree_cpu[active].mean().item())
        else:
            stats["avg_fg_degree"] = 0.0
    else:
        stats["edges_total"] = 0.0
        stats["avg_degree"] = 0.0
        stats["avg_fg_degree"] = 0.0

    degree = torch.clamp(degree, min=1e-6)
    normalized_weights = edge_wgt_t / degree[edge_tgt_t]

    # convert stats to floats for logging
    stats_out = {k: float(v) for k, v in stats.items()}

    return edge_index, normalized_weights, stats_out
