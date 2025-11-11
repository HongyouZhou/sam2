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
    spatially or semantically related objects. Visual features are projected to style_dim
    via MLP and fused with style deltas for content-aware refinement.

    Args:
        style_dim: Dimension of style perturbation vector (6 for Gram matrix style)
        feature_dim: Dimension of visual features (e.g., 256 for backbone_fpn[-1]).
                     If > 0, creates a feature projection MLP and fuses with styles in GCN
        num_layers: Number of GCN layers (default: 2)
    
    Architecture:
        1. Feature Projection MLP (if feature_dim > 0):
           feature_dim (256) → feature_dim/2 (128) → style_dim (6)
           
        2. Node Feature Fusion:
           style_deltas [B, K, 6] + projected_features [B, K, 6] → [B, K, 12]
           
        3. GCN Layers (operate on fused features):
           - Intermediate layers: [12] → [12] with ReLU + LayerNorm
           - Last layer: [12] → [6] (project back to style space)
           
        This ensures features and styles have equal weight (both 6-dim) before fusion.
    """

    def __init__(self, style_dim: int = 6, feature_dim: int = 0, num_layers: int = 2):
        super().__init__()

        self.style_dim = style_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # Node dimension: style + projected features (both style_dim)
        self.node_dim = style_dim * 2 if feature_dim > 0 else style_dim

        # Feature projection MLP: project high-dim features to style_dim
        # This balances the contribution of features and styles (both become 6-dim)
        if feature_dim > 0:
            self.feature_projection = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, style_dim),
            )
            # Initialize with small weights for stable projection
            for m in self.feature_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(m.bias)
        else:
            self.feature_projection = None

        # Build GCN layers
        # Intermediate layers maintain node_dim, last layer projects to style_dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                # Intermediate layers: maintain node_dim
                layer = nn.Linear(self.node_dim, self.node_dim)
            else:
                # Last layer: project back to style_dim only
                layer = nn.Linear(self.node_dim, style_dim)
            
            # Initialize all layers with small weights for near-identity behavior
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)
            
            self.layers.append(layer)

        # Layer normalization for stability
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.layer_norms.append(nn.LayerNorm(self.node_dim))
            else:
                self.layer_norms.append(nn.LayerNorm(style_dim))
        
        self.dropout = 0.1

    def forward(self, style_deltas: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                mask_features: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of GCN for refining style perturbations."""
        B, num_nodes, _ = style_deltas.shape

        # Fuse visual features with style deltas if provided
        if mask_features is not None and self.feature_projection is not None:
            projected_features = self.project_features(mask_features)
            node_features = torch.cat([style_deltas, projected_features], dim=-1)
        else:
            node_features = style_deltas

        x = node_features.reshape(B * num_nodes, node_features.shape[-1])

        # Apply GCN layers: Transform -> Aggregate -> Activate
        for layer_idx, (linear, norm) in enumerate(zip(self.layers, self.layer_norms)):
            x = linear(x)
            x = self._graph_conv(x, edge_index, edge_weight, B, num_nodes)
            
            if layer_idx < self.num_layers - 1:
                x = norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.reshape(B, num_nodes, self.style_dim)
        return x
    
    def project_features(self, mask_features: torch.Tensor) -> torch.Tensor:
        """Project high-dim features to style_dim for balanced fusion."""
        if self.feature_projection is None:
            raise RuntimeError("Feature projection not available. Set feature_dim > 0.")
        
        B, K, C = mask_features.shape
        features_flat = mask_features.reshape(B * K, C)
        projected = self.feature_projection(features_flat)
        projected = projected.reshape(B, K, self.style_dim)
        return projected
    
    def _add_self_loops(
        self, 
        edge_index: torch.Tensor, 
        edge_weight: torch.Tensor, 
        num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add self-loops to preserve node identity."""
        device = edge_index.device
        
        loop_index = torch.arange(num_nodes, dtype=torch.long, device=device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(num_nodes, dtype=edge_weight.dtype, device=device)
        
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        
        return edge_index, edge_weight

    def _graph_conv(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, 
                    batch_size: int, num_nodes_per_batch: int) -> torch.Tensor:
        """Graph convolution: aggregate neighbor features with weights."""
        if edge_index.numel() == 0:
            return x

        src, tgt = edge_index[0], edge_index[1]
        out = x.new_zeros(x.size(0), x.size(1))
        
        src_features = x[src]
        weighted_features = src_features * edge_weight.unsqueeze(1)
        out.index_add_(0, tgt, weighted_features)
        
        return out


def compute_boundary_distance(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    grid_y: torch.Tensor,
    grid_x: torch.Tensor,
) -> float:
    """Compute minimum distance between boundaries of two masks."""
    kernel_size = 3
    padding = kernel_size // 2

    mask1_expanded = mask1.unsqueeze(0).unsqueeze(0)
    mask2_expanded = mask2.unsqueeze(0).unsqueeze(0)

    mask1_dilated = F.max_pool2d(mask1_expanded, kernel_size=kernel_size, stride=1, padding=padding)
    mask2_dilated = F.max_pool2d(mask2_expanded, kernel_size=kernel_size, stride=1, padding=padding)

    boundary1 = (mask1_dilated - mask1_expanded).squeeze() > 0.5
    boundary2 = (mask2_dilated - mask2_expanded).squeeze() > 0.5

    if not boundary1.any() or not boundary2.any():
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

    boundary1_coords = torch.stack([grid_y[0, 0][boundary1], grid_x[0, 0][boundary1]], dim=1)
    boundary2_coords = torch.stack([grid_y[0, 0][boundary2], grid_x[0, 0][boundary2]], dim=1)

    if boundary1_coords.shape[0] == 0 or boundary2_coords.shape[0] == 0:
        return float("inf")

    diff = boundary1_coords.unsqueeze(1) - boundary2_coords.unsqueeze(0)
    distances = (diff**2).sum(dim=2) ** 0.5
    
    return distances.min().item()


def build_object_graph(
    masks: torch.Tensor,
    img_batch: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    edge_threshold: float = 0.3,
    use_semantic: bool = True,
    use_background: bool = False,
    distance_threshold: float | None = None,
    use_boundary_distance: bool = False,
    mask_features: Optional[torch.Tensor] = None,
    feature_sim_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build object graph from masks for GCN message passing."""
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
        "edges_semantic": 0,
    }
    
    # Pre-compute normalized features for semantic similarity if provided
    normalized_features = None
    if mask_features is not None:
        # L2 normalize features for cosine similarity
        normalized_features = F.normalize(mask_features, p=2, dim=-1)  # [B, K, C]

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

        # Semantic edges based on visual feature similarity
        if normalized_features is not None and len(object_indices) > 1:
            for i_pos, i in enumerate(object_indices):
                feat_i = normalized_features[b, i]  # [C]
                node_i = b * nodes_per_batch + i
                
                for j in object_indices[i_pos + 1:]:
                    feat_j = normalized_features[b, j]  # [C]
                    
                    # Cosine similarity (dot product of normalized vectors)
                    similarity = (feat_i * feat_j).sum().item()
                    
                    if similarity < feature_sim_threshold:
                        continue
                    
                    # Debug log for first semantic edge found in first batch
                    if b == 0 and stats["edges_semantic"] == 0:
                        import logging
                        logging.debug(f"GCN: First semantic edge found: obj_{i} <-> obj_{j}, similarity={similarity:.3f}, threshold={feature_sim_threshold}")
                    
                    # Use similarity as edge weight
                    node_j = b * nodes_per_batch + j
                    edge_src.extend([node_i, node_j])
                    edge_tgt.extend([node_j, node_i])
                    edge_wgt.extend([similarity, similarity])
                    stats["edges_semantic"] += 2
        
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
                "edges_semantic": 0.0,
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
