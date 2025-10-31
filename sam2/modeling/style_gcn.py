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
            style_deltas: Style perturbations [B, K, style_dim]
            edge_index: Edge indices [2, E] where E is number of edges
            edge_weight: Edge weights [E]
            
        Returns:
            Refined style perturbations [B, K, style_dim]
        """
        B, K, D = style_deltas.shape
        
        # Flatten batch and nodes for processing
        x = style_deltas.reshape(B * K, D)  # [B*K, D]
        
        # Apply GCN layers (now all maintain same dimension for residual connections)
        for layer_idx, (linear, norm, alpha) in enumerate(
            zip(self.layers, self.layer_norms, self.alphas)
        ):
            x_input = x  # Save input for residual connection
            
            # Message passing
            x = self._graph_conv(x, edge_index, edge_weight, B, K)
            
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
        
        # Reshape back to [B, K, D]
        x = x.reshape(B, K, D)
        
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
        Perform graph convolution operation.
        
        Args:
            x: Node features [B*K, D]
            edge_index: Edge indices [2, E]
            edge_weight: Edge weights [E]
            batch_size: Batch size B
            num_nodes_per_batch: Number of nodes per batch K
            
        Returns:
            Aggregated features [B*K, D]
        """
        if edge_index.numel() == 0:
            # No edges, return identity
            return x
        
        # Extract source and target nodes
        src, tgt = edge_index[0], edge_index[1]
        
        # Gather source features
        src_features = x[src]  # [E, D]
        
        # Weight by edge weights
        weighted_features = src_features * edge_weight.unsqueeze(1)  # [E, D]
        
        # Aggregate to target nodes
        out = torch.zeros_like(x)
        out.index_add_(0, tgt, weighted_features)
        
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
    
    edge_list = []
    edge_weights = []
    
    # Process each batch separately
    for b in range(B):
        batch_edges = []
        batch_weights = []
        
        # Compute spatial edges based on IoU
        for i in range(K):
            for j in range(i + 1, K):
                mask_i = masks_binary[b, i]
                mask_j = masks_binary[b, j]
                
                # Compute IoU
                intersection = (mask_i * mask_j).sum()
                union = mask_i.sum() + mask_j.sum() - intersection
                
                if union > 0:
                    iou = (intersection / union).item()
                    
                    if iou > edge_threshold:
                        # Add bidirectional edge
                        node_i = b * K + i
                        node_j = b * K + j
                        batch_edges.extend([[node_i, node_j], [node_j, node_i]])
                        batch_weights.extend([iou, iou])
        
        # Compute semantic edges (same label)
        if use_semantic and labels is not None:
            batch_labels = labels[b]  # [K]
            for i in range(K):
                for j in range(i + 1, K):
                    if batch_labels[i] == batch_labels[j] and batch_labels[i] >= 0:
                        node_i = b * K + i
                        node_j = b * K + j
                        # Check if edge already exists
                        if [node_i, node_j] not in batch_edges:
                            # Add semantic edge with weight 1.0
                            batch_edges.extend([[node_i, node_j], [node_j, node_i]])
                            batch_weights.extend([1.0, 1.0])
        
        # Add background edges
        if use_background:
            background_node = b * K + K  # Background as K+1-th node
            for i in range(K):
                node_i = b * K + i
                batch_edges.extend([[node_i, background_node], [background_node, node_i]])
                batch_weights.extend([0.5, 0.5])  # Fixed weight for background
        
        edge_list.extend(batch_edges)
        edge_weights.extend(batch_weights)
    
    if len(edge_list) == 0:
        # No edges, return empty tensors
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device)
        )
    
    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()  # [2, E]
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=device)  # [E]
    
    # Normalize edge weights by degree
    num_nodes = B * K
    degree = torch.zeros(num_nodes, device=device)
    degree.index_add_(0, edge_index[1], edge_weight)
    
    # Avoid division by zero
    degree = torch.clamp(degree, min=1.0)
    
    # Normalize edge weights
    normalized_weights = edge_weight / degree[edge_index[1]]
    
    return edge_index, normalized_weights

