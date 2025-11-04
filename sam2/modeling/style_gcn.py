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
    spatially or semantically related objects.
    
    Args:
        style_dim: Dimension of style perturbation vector (6 for Gram matrix style)
        hidden_dim: Hidden dimension for GCN layers
        num_layers: Number of GCN layers (default: 2)
    """
    
    def __init__(self, style_dim: int = 6, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build GCN layers (all layers keep same dimension for residual connections)
        # Using style_dim throughout allows per-layer residual connections
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Linear(style_dim, style_dim)
            
            # Initialize all layers with small weights for near-identity behavior
            # Combined with alpha ≈ 0, this makes the GCN output close to input initially
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # Small random init
            nn.init.zeros_(layer.bias)
            
            self.layers.append(layer)
        
        # Layer normalization for stability (all layers use style_dim)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(style_dim)
            for i in range(num_layers)
        ])
        
        # Learnable residual weights (alpha) per layer
        # Initialize to -10.0 for very strong near-identity mapping at start
        # sigmoid(-10) ≈ 0.000045, so output ≈ 0.00005 * GCN(x) + 0.99995 * x ≈ x
        # This ensures GCN doesn't disrupt PGD-found adversarial samples initially
        # and allows gradual learning during training
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor([-10.0])) for _ in range(num_layers)
        ])
        
    def forward(
        self, 
        style_deltas: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of GCN.
        
        Args:
            style_deltas: Style perturbations [B, num_nodes, style_dim]
                         where num_nodes = K (foreground only) or K+1 (with background)
            edge_index: Edge indices [2, E] where E is number of edges
            edge_weight: Edge weights [E]
            
        Returns:
            Refined style perturbations [B, num_nodes, style_dim]
        """
        B, num_nodes, D = style_deltas.shape
        
        # Flatten batch and nodes for processing
        x = style_deltas.reshape(B * num_nodes, D)  # [B*num_nodes, D]
        
        # Apply GCN layers (now all maintain same dimension for residual connections)
        for layer_idx, (linear, norm, alpha) in enumerate(
            zip(self.layers, self.layer_norms, self.alphas)
        ):
            x_input = x  # Save input for residual connection
            
            # Message passing (use actual num_nodes from input shape)
            x = self._graph_conv(x, edge_index, edge_weight, B, num_nodes)
            
            # Linear transformation
            x = linear(x)
            
            # Layer normalization
            x = norm(x)
            
            # ReLU activation (except last layer)
            if layer_idx < self.num_layers - 1:
                x = F.relu(x)
            
            # Residual connection with learnable weight (works for all layers now)
            alpha_val = torch.sigmoid(alpha)  # Ensure alpha in [0, 1]
            x = alpha_val * x + (1 - alpha_val) * x_input
        
        # Reshape back to [B, num_nodes, D]
        x = x.reshape(B, num_nodes, D)
        
        return x
    
    def _graph_conv(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weight: torch.Tensor,
        batch_size: int,
        num_nodes_per_batch: int
    ) -> torch.Tensor:
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


def build_object_graph(
    masks: torch.Tensor,
    img_batch: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    edge_threshold: float = 0.3,
    use_semantic: bool = True,
    use_background: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build object graph from masks for GCN message passing.
    
    Constructs a graph where:
    - Nodes are objects (and optionally background)
    - Edges connect spatially close or semantically similar objects
    - Edge weights are normalized by degree
    
    Args:
        masks: Binary masks [B, K, H, W] or [B, K, 1, H, W]
        img_batch: Optional images (not used currently, for future semantic features)
        labels: Optional object labels [B, K] for semantic edges
        edge_threshold: IoU threshold for spatial edges
        use_semantic: Whether to add semantic edges between same-label objects
        use_background: Whether to include background as node K+1
        
    Returns:
        edge_index: Edge indices [2, E] in COO format
        edge_weight: Edge weights [E], normalized by degree
    """
    # Handle 5D masks [B, K, 1, H, W] -> [B, K, H, W]
    if masks.ndim == 5:
        masks = masks.squeeze(2)
    
    B, K, H, W = masks.shape
    device = masks.device
    
    # Binarize masks
    masks_binary = (masks > 0.5).float()
    
    # IMPORTANT: When use_background=True, it means the last mask (index K-1) IS the background.
    # We don't add an extra node; the background is already in the masks.
    # So nodes_per_batch = K regardless of use_background flag.
    nodes_per_batch = K
    
    # Pre-allocate maximum possible edges
    # Spatial: B * K*(K-1) bidirectional edges
    # Background: B * (K-1) * 2 edges if enabled (connecting K-1 objects to 1 background)
    max_spatial_edges = B * K * (K - 1)
    max_bg_edges = B * (K - 1) * 2 if use_background else 0
    max_edges = max_spatial_edges + max_bg_edges
    
    # Pre-allocate tensors on device
    edge_src = torch.zeros(max_edges, dtype=torch.long, device=device)
    edge_tgt = torch.zeros(max_edges, dtype=torch.long, device=device)
    edge_wgt = torch.zeros(max_edges, dtype=torch.float32, device=device)
    edge_count = 0
    
    # Process each batch separately
    for b in range(B):
        # Compute spatial edges based on IoU (vectorized within batch)
        for i in range(K):
            mask_i = masks_binary[b, i]
            area_i = mask_i.sum()
            
            for j in range(i + 1, K):
                mask_j = masks_binary[b, j]
                area_j = mask_j.sum()
                
                # Compute IoU
                intersection = (mask_i * mask_j).sum()
                union = area_i + area_j - intersection
                
                if union > 0:
                    iou = intersection / union
                    
                    if iou > edge_threshold:
                        # Add bidirectional edge (use nodes_per_batch for correct indexing)
                        node_i = b * nodes_per_batch + i
                        node_j = b * nodes_per_batch + j
                        
                        # Add edge i->j
                        edge_src[edge_count] = node_i
                        edge_tgt[edge_count] = node_j
                        edge_wgt[edge_count] = iou
                        edge_count += 1
                        
                        # Add edge j->i
                        edge_src[edge_count] = node_j
                        edge_tgt[edge_count] = node_i
                        edge_wgt[edge_count] = iou
                        edge_count += 1
        
        # Compute semantic edges (same label) - skipped for now as labels are None
        
        # Add background edges
        if use_background:
            # Background node is the last mask (index K-1) in each batch
            # It's already included in the K masks, so we connect it to other objects
            background_node = b * nodes_per_batch + (K - 1)
            # Connect all non-background objects (0 to K-2) to background
            for i in range(K - 1):
                node_i = b * nodes_per_batch + i
                
                # Edge object->background
                edge_src[edge_count] = node_i
                edge_tgt[edge_count] = background_node
                edge_wgt[edge_count] = 0.5
                edge_count += 1
                
                # Edge background->object
                edge_src[edge_count] = background_node
                edge_tgt[edge_count] = node_i
                edge_wgt[edge_count] = 0.5
                edge_count += 1
    
    if edge_count == 0:
        # No edges, return empty tensors
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device)
        )
    
    # Trim to actual edge count
    edge_src = edge_src[:edge_count]
    edge_tgt = edge_tgt[:edge_count]
    edge_wgt = edge_wgt[:edge_count]
    
    # Stack into edge_index
    edge_index = torch.stack([edge_src, edge_tgt], dim=0)  # [2, E]
    
    # Normalize edge weights by degree
    # Total nodes = B * K (background is already included in K if use_background=True)
    num_nodes = B * K
    degree = torch.zeros(num_nodes, device=device)
    degree.index_add_(0, edge_tgt, edge_wgt)
    
    # Avoid division by zero
    degree = torch.clamp(degree, min=1.0)
    
    # Normalize edge weights
    normalized_weights = edge_wgt / degree[edge_tgt]
    
    return edge_index, normalized_weights

