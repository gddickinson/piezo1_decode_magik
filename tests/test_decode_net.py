"""
Smoke tests for DECODE network forward-pass shapes.
"""

import pytest
import torch
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.decode_net import DECODENet, FrameAnalysisNet, TemporalContextNet


class TestFrameAnalysisNet:
    """Tests for the FrameAnalysisNet sub-network."""

    def test_output_shape(self):
        """Forward pass produces correct output shape."""
        model = FrameAnalysisNet(base_channels=16)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 16, 64, 64)

    def test_different_base_channels(self):
        """Works with non-default base_channels."""
        model = FrameAnalysisNet(base_channels=8)
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert out.shape == (1, 8, 32, 32)


class TestTemporalContextNet:
    """Tests for the TemporalContextNet sub-network."""

    def test_output_shape(self):
        """Forward pass produces correct output shape."""
        base_ch = 16
        model = TemporalContextNet(base_channels=base_ch)
        x = torch.randn(2, 3 * base_ch, 64, 64)
        out = model(x)
        assert out.shape == (2, base_ch, 64, 64)


class TestDECODENet:
    """Tests for the full DECODE network."""

    @pytest.fixture
    def model(self):
        return DECODENet(base_channels=16)

    def test_forward_pass_shapes(self, model):
        """All output heads produce correct shapes."""
        x = torch.randn(2, 3, 64, 64)
        outputs = model(x)

        assert outputs['prob'].shape == (2, 1, 64, 64)
        assert outputs['offset'].shape == (2, 2, 64, 64)
        assert outputs['photons'].shape == (2, 1, 64, 64)
        assert outputs['uncertainty'].shape == (2, 3, 64, 64)

    def test_prob_range(self, model):
        """Detection probability is in [0, 1] (sigmoid output)."""
        x = torch.randn(2, 3, 64, 64)
        outputs = model(x)
        assert outputs['prob'].min() >= 0.0
        assert outputs['prob'].max() <= 1.0

    def test_offset_range(self, model):
        """Sub-pixel offsets are in [-0.5, 0.5] (tanh * 0.5)."""
        x = torch.randn(2, 3, 64, 64)
        outputs = model(x)
        assert outputs['offset'].min() >= -0.5
        assert outputs['offset'].max() <= 0.5

    def test_photons_non_negative(self, model):
        """Photon counts are non-negative (softplus output)."""
        x = torch.randn(2, 3, 64, 64)
        outputs = model(x)
        assert outputs['photons'].min() >= 0.0

    def test_requires_3_frames(self, model):
        """Raises assertion for wrong number of frames."""
        x = torch.randn(2, 2, 64, 64)
        with pytest.raises(AssertionError):
            model(x)

    def test_batch_size_one(self, model):
        """Works with batch size 1."""
        x = torch.randn(1, 3, 64, 64)
        outputs = model(x)
        assert outputs['prob'].shape[0] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
