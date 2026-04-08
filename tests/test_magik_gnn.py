"""
Smoke tests for MAGIK GNN forward-pass shapes.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.magik_gnn import MAGIKNet, EdgeConvLayer, build_tracking_graph


class TestEdgeConvLayer:
    """Tests for the EdgeConv layer."""

    def test_output_shape(self):
        """Forward pass produces correct output shape."""
        layer = EdgeConvLayer(in_channels=16, out_channels=32)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (10, 32)

    def test_no_edges(self):
        """Works with no edges."""
        layer = EdgeConvLayer(in_channels=16, out_channels=32)
        x = torch.randn(5, 16)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (5, 32)
        assert (out == 0).all()


class TestMAGIKNet:
    """Tests for the full MAGIK network."""

    @pytest.fixture
    def model(self):
        return MAGIKNet(node_features=5, edge_features=4,
                        hidden_dim=32, num_layers=2)

    def test_forward_pass_shape(self, model):
        """Edge logits shape matches number of edges."""
        N, E = 20, 50
        node_features = torch.randn(N, 5)
        edge_index = torch.randint(0, N, (2, E))
        edge_features = torch.randn(E, 4)

        logits = model(node_features, edge_index, edge_features)
        assert logits.shape == (E,)

    def test_no_edges(self, model):
        """Returns empty tensor when no edges."""
        node_features = torch.randn(5, 5)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_features = torch.zeros((0, 4))

        logits = model(node_features, edge_index, edge_features)
        assert logits.shape == (0,)

    def test_sigmoid_output_range(self, model):
        """Sigmoid of logits is in [0, 1]."""
        N, E = 10, 30
        node_features = torch.randn(N, 5)
        edge_index = torch.randint(0, N, (2, E))
        edge_features = torch.randn(E, 4)

        model.eval()
        with torch.no_grad():
            logits = model(node_features, edge_index, edge_features)
            probs = torch.sigmoid(logits)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


class TestBuildTrackingGraph:
    """Tests for the graph building utility."""

    def test_empty_detections(self):
        """Returns empty tensors for no detections."""
        nf, ei, ef = build_tracking_graph([])
        assert nf.shape == (0, 5)
        assert ei.shape == (2, 0)
        assert ef.shape == (0, 4)

    def test_basic_graph(self):
        """Builds graph from simple detections."""
        detections = [
            {'frame': 0, 'x': 10.0, 'y': 10.0, 'prob': 0.9, 'photons': 1000},
            {'frame': 1, 'x': 11.0, 'y': 10.0, 'prob': 0.8, 'photons': 900},
        ]
        nf, ei, ef = build_tracking_graph(detections,
                                           max_temporal_gap=1,
                                           max_spatial_distance=5)
        assert nf.shape == (2, 5)
        assert ei.shape[0] == 2  # [src, dst]
        assert ei.shape[1] >= 1  # at least one edge
        assert ef.shape[1] == 4  # dx, dy, distance, temporal_gap

    def test_distance_filtering(self):
        """Detections far apart are not connected."""
        detections = [
            {'frame': 0, 'x': 0.0, 'y': 0.0},
            {'frame': 1, 'x': 100.0, 'y': 100.0},
        ]
        nf, ei, ef = build_tracking_graph(detections,
                                           max_temporal_gap=1,
                                           max_spatial_distance=5)
        assert ei.shape[1] == 0  # no edges


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
