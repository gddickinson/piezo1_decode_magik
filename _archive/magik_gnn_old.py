"""
MAGIK-style Graph Neural Network for Particle Tracking

Based on graph neural networks where:
- Nodes = detected puncta
- Edges = potential links between frames
- GNN learns edge probabilities
- Linking via global optimization

Architecture inspired by:
- MAGIK (DeepTrack 2.0)
- Trackastra (Transformer-based tracking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops


class EdgeConv(MessagePassing):
    """
    Edge convolution layer for learning edge features.
    
    Aggregates features from neighboring nodes.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')  # Max aggregation
        
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
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge connections
        Returns:
            out: (N, out_channels) updated node features
        """
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        """
        Create messages from neighboring nodes.
        
        Args:
            x_i: Features of target nodes
            x_j: Features of source nodes
        Returns:
            messages: Updated edge features
        """
        # Concatenate features
        edge_features = torch.cat([x_i, x_j], dim=1)
        
        # Apply MLP
        return self.mlp(edge_features)


class MAGIKNet(nn.Module):
    """
    Graph Neural Network for particle linking.
    
    Learns to predict which detections should be linked across frames.
    """
    
    def __init__(self,
                 node_features=6,  # x, y, t, intensity, σx, σy
                 edge_features=4,  # Δx, Δy, Δt, distance
                 hidden_dim=128,
                 num_layers=3):
        super().__init__()
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            EdgeConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Edge classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)  # Binary: link or not
        )
    
    def forward(self, node_features, edge_index, edge_features):
        """
        Forward pass.
        
        Args:
            node_features: (N, node_features) node attributes
            edge_index: (2, E) edge connectivity
            edge_features: (E, edge_features) edge attributes
            
        Returns:
            edge_probs: (E,) probability that each edge is a link
        """
        
        # Encode node features
        x = self.node_encoder(node_features)  # (N, hidden_dim)
        
        # Encode edge features
        edge_attr = self.edge_encoder(edge_features)  # (E, hidden_dim)
        
        # Graph convolutions (message passing)
        for conv in self.conv_layers:
            x = conv(x, edge_index) + x  # Residual connection
        
        # For each edge, concatenate source/target node features + edge features
        src, dst = edge_index
        edge_input = torch.cat([
            x[src],
            x[dst],
            edge_features
        ], dim=1)  # (E, 2*hidden_dim + edge_features)
        
        # Classify edges
        logits = self.edge_classifier(edge_input)  # (E, 1)
        probs = torch.sigmoid(logits.squeeze())  # (E,)
        
        return probs
    
    def predict_links(self, node_features, edge_index, edge_features, threshold=0.5):
        """
        Predict which edges are actual links.
        
        Args:
            node_features: Node features
            edge_index: Candidate edges
            edge_features: Edge features
            threshold: Probability threshold
            
        Returns:
            links: (M, 2) array of linked node pairs
            probs: (M,) probabilities for each link
        """
        probs = self.forward(node_features, edge_index, edge_features)
        
        # Select edges above threshold
        mask = probs > threshold
        links = edge_index[:, mask].t()
        link_probs = probs[mask]
        
        return links.cpu().numpy(), link_probs.cpu().numpy()


def build_tracking_graph(detections_per_frame, max_frame_gap=5, max_distance=10.0):
    """
    Build tracking graph from detections.
    
    Args:
        detections_per_frame: List[List[Dict]] - detections per frame
            Each detection: {'x', 'y', 'photons', 'sigma_x', 'sigma_y'}
        max_frame_gap: Maximum frames to link across
        max_distance: Maximum spatial distance to consider
        
    Returns:
        node_features: (N, 6) tensor [x, y, t, intensity, σx, σy]
        edge_index: (2, E) tensor
        edge_features: (E, 4) tensor [Δx, Δy, Δt, distance]
        node_to_detection: List mapping node index to (frame, det_idx)
    """
    
    # Build nodes
    node_features = []
    node_to_detection = []
    
    for frame_idx, detections in enumerate(detections_per_frame):
        for det_idx, det in enumerate(detections):
            node_features.append([
                det['x'],
                det['y'],
                float(frame_idx),
                det['photons'],
                det['sigma_x'],
                det['sigma_y']
            ])
            node_to_detection.append((frame_idx, det_idx))
    
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    # Build edges (candidate links)
    edges = []
    edge_features_list = []
    
    num_nodes = len(node_features)
    
    for i in range(num_nodes):
        frame_i, _ = node_to_detection[i]
        x_i, y_i = node_features[i, 0], node_features[i, 1]
        
        for j in range(i + 1, num_nodes):
            frame_j, _ = node_to_detection[j]
            
            # Only link forward in time within max_frame_gap
            frame_gap = frame_j - frame_i
            if frame_gap <= 0 or frame_gap > max_frame_gap:
                continue
            
            x_j, y_j = node_features[j, 0], node_features[j, 1]
            
            # Compute distance
            dx = x_j - x_i
            dy = y_j - y_i
            distance = torch.sqrt(dx**2 + dy**2)
            
            # Only consider nearby particles
            if distance > max_distance:
                continue
            
            # Add edge
            edges.append([i, j])
            edge_features_list.append([
                dx.item(),
                dy.item(),
                float(frame_gap),
                distance.item()
            ])
    
    if len(edges) == 0:
        # No edges - return empty graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_features = torch.zeros((0, 4), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_features = torch.tensor(edge_features_list, dtype=torch.float32)
    
    return node_features, edge_index, edge_features, node_to_detection


# Test the GNN
if __name__ == '__main__':
    import numpy as np
    
    print("Testing MAGIK GNN...")
    
    # Create model
    model = MAGIKNet(
        node_features=6,
        edge_features=4,
        hidden_dim=128,
        num_layers=3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy detections
    detections_per_frame = []
    for t in range(10):
        frame_detections = []
        for i in range(5):  # 5 puncta per frame
            # Random walk
            x = 128 + i * 30 + np.random.randn() * 2
            y = 128 + i * 30 + np.random.randn() * 2
            
            frame_detections.append({
                'x': x,
                'y': y,
                'photons': 1000 + np.random.randn() * 100,
                'sigma_x': 1.0,
                'sigma_y': 1.0
            })
        detections_per_frame.append(frame_detections)
    
    # Build graph
    node_features, edge_index, edge_features, node_map = build_tracking_graph(
        detections_per_frame,
        max_frame_gap=3,
        max_distance=10.0
    )
    
    print(f"\n✅ Graph built:")
    print(f"  Nodes: {node_features.shape[0]}")
    print(f"  Edges: {edge_index.shape[1]}")
    
    # Forward pass
    edge_probs = model(node_features, edge_index, edge_features)
    
    print(f"\n✅ Forward pass successful")
    print(f"  Edge probabilities shape: {edge_probs.shape}")
    print(f"  Mean probability: {edge_probs.mean():.3f}")
    
    # Predict links
    links, probs = model.predict_links(
        node_features, edge_index, edge_features, threshold=0.5
    )
    
    print(f"\n✅ Prediction successful")
    print(f"  Predicted links: {len(links)}")
    
    print("\n✅ All tests passed!")
