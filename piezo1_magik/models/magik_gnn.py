"""
MAGIK Graph Neural Network for Particle Tracking

Implements EdgeConv-style GNN for learning to link detections into tracks.
No external dependencies beyond PyTorch!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConvLayer(nn.Module):
    """
    EdgeConv layer for graph neural networks.
    
    Implements message passing along edges without torch_geometric dependency.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # MLP for edge features
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge connections [source, target]
            
        Returns:
            x_out: (N, out_channels) updated node features
        """
        src, dst = edge_index  # (E,), (E,)
        
        # Get source and destination features
        x_src = x[src]  # (E, in_channels)
        x_dst = x[dst]  # (E, in_channels)
        
        # Concatenate for each edge
        edge_features = torch.cat([x_src, x_dst], dim=1)  # (E, 2*in_channels)
        
        # Apply MLP
        edge_updates = self.mlp(edge_features)  # (E, out_channels)
        
        # Aggregate to destination nodes using scatter_add (sum aggregation)
        # This is more stable than max for gradient computation
        x_out = torch.zeros(x.shape[0], edge_updates.shape[1], 
                           device=x.device, dtype=x.dtype)
        
        # Use index_add instead of in-place loop (MPS compatible)
        x_out.index_add_(0, dst, edge_updates)
        
        return x_out


class MAGIKNet(nn.Module):
    """
    MAGIK Graph Neural Network for particle tracking.
    
    Predicts which edges represent true particle links.
    """
    
    def __init__(self, node_features=5, edge_features=4, 
                 hidden_dim=64, num_layers=3):
        """
        Initialize MAGIK.
        
        Args:
            node_features: Number of node features (frame, x, y, prob, photons)
            edge_features: Number of edge features (dx, dy, distance, temporal_gap)
            hidden_dim: Hidden dimension
            num_layers: Number of EdgeConv layers
        """
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Edge feature embedding
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # EdgeConv layers
        self.conv_layers = nn.ModuleList([
            EdgeConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Edge classifier
        # Takes: [node_src, node_dst, edge_features, difference]
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)  # Output logit (before sigmoid)
        )
    
    def forward(self, node_features, edge_index, edge_features):
        """
        Forward pass.
        
        Args:
            node_features: (N, node_features) detection features
            edge_index: (2, E) edge connections
            edge_features: (E, edge_features) edge features
            
        Returns:
            edge_logits: (E,) predicted edge logits (apply sigmoid for probabilities)
        """
        # Encode nodes
        x = self.node_encoder(node_features)  # (N, hidden_dim)
        
        # Encode edges (only if we have edges)
        if edge_features.shape[0] > 0:
            edge_emb = self.edge_encoder(edge_features)  # (E, hidden_dim)
        else:
            edge_emb = torch.zeros((0, self.hidden_dim), device=node_features.device)
        
        # Apply EdgeConv layers
        for conv in self.conv_layers:
            if edge_index.shape[1] > 0:  # Only if we have edges
                x_update = conv(x, edge_index)
                x = x + x_update  # Residual connection
                x = F.relu(x)
        
        # Get edge predictions
        if edge_index.shape[1] == 0:
            return torch.zeros(0, device=node_features.device)
        
        src, dst = edge_index
        x_src = x[src]  # (E, hidden_dim)
        x_dst = x[dst]  # (E, hidden_dim)
        
        # Concatenate all edge information
        edge_input = torch.cat([x_src, x_dst, edge_features], dim=1)  # (E, 2*hidden_dim + edge_features)
        
        # Predict edge probabilities
        edge_logits = self.edge_classifier(edge_input).squeeze(-1)  # (E,)
        
        return edge_logits


def build_tracking_graph(detections, max_temporal_gap=2, max_spatial_distance=30):
    """
    Build temporal graph from detections.
    
    Args:
        detections: List of dicts with keys: frame, x, y, prob, photons
        max_temporal_gap: Maximum frame gap for edges
        max_spatial_distance: Maximum spatial distance for edges
        
    Returns:
        node_features: (N, 5) tensor
        edge_index: (2, E) tensor
        edge_features: (E, 4) tensor
    """
    if len(detections) == 0:
        return (torch.zeros((0, 5)), 
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 4)))
    
    # Create node features
    N = len(detections)
    node_features = torch.zeros((N, 5))
    
    for i, det in enumerate(detections):
        node_features[i] = torch.tensor([
            det['frame'],
            det['x'],
            det['y'],
            det.get('prob', 1.0),
            det.get('photons', 1000.0)
        ])
    
    # Build edges
    edge_list = []
    edge_feat_list = []
    
    # Group by frame
    frame_to_nodes = {}
    for i, det in enumerate(detections):
        frame = det['frame']
        if frame not in frame_to_nodes:
            frame_to_nodes[frame] = []
        frame_to_nodes[frame].append(i)
    
    frames = sorted(frame_to_nodes.keys())
    
    # Connect nearby frames
    for i, frame in enumerate(frames):
        for j in range(i + 1, min(i + 1 + max_temporal_gap, len(frames))):
            next_frame = frames[j]
            temporal_gap = next_frame - frame
            
            if temporal_gap > max_temporal_gap:
                break
            
            # Connect all pairs within distance
            for src_node in frame_to_nodes[frame]:
                src_x = detections[src_node]['x']
                src_y = detections[src_node]['y']
                
                for dst_node in frame_to_nodes[next_frame]:
                    dst_x = detections[dst_node]['x']
                    dst_y = detections[dst_node]['y']
                    
                    dx = dst_x - src_x
                    dy = dst_y - src_y
                    distance = (dx**2 + dy**2)**0.5
                    
                    if distance <= max_spatial_distance:
                        edge_list.append([src_node, dst_node])
                        edge_feat_list.append([dx, dy, distance, temporal_gap])
    
    if len(edge_list) == 0:
        return (node_features,
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 4)))
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_features = torch.tensor(edge_feat_list, dtype=torch.float32)
    
    return node_features, edge_index, edge_features


# Test
if __name__ == '__main__':
    import numpy as np
    
    print("Testing MAGIK GNN (no torch_geometric)...")
    
    # Create model
    model = MAGIKNet(
        node_features=5,
        edge_features=4,
        hidden_dim=64,
        num_layers=3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy detections
    detections = []
    for t in range(10):
        for i in range(5):  # 5 particles
            x = 100 + i * 30 + np.random.randn() * 2
            y = 100 + i * 30 + np.random.randn() * 2
            detections.append({
                'frame': t,
                'x': x,
                'y': y,
                'prob': 0.9,
                'photons': 1000
            })
    
    # Build graph
    node_features, edge_index, edge_features = build_tracking_graph(
        detections, max_temporal_gap=2, max_spatial_distance=30
    )
    
    print(f"\n✅ Graph built:")
    print(f"  Nodes: {node_features.shape[0]}")
    print(f"  Edges: {edge_index.shape[1]}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        edge_logits = model(node_features, edge_index, edge_features)
        edge_probs = torch.sigmoid(edge_logits)
    
    print(f"\n✅ Forward pass successful")
    print(f"  Edge probabilities shape: {edge_probs.shape}")
    print(f"  Mean probability: {edge_probs.mean():.3f}")
    
    print("\n✅ All tests passed!")
