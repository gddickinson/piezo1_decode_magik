"""
DECODE Network for Puncta Localization

Based on: Speiser et al., "Deep learning enables fast and dense 
single-molecule localization with high accuracy", Nature Methods 2021

Architecture:
- Processes 3-frame temporal windows
- Two stacked U-Nets (Frame Analysis + Temporal Context)
- Outputs: Detection probability, coordinates, photon counts, uncertainties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameAnalysisNet(nn.Module):
    """
    First U-Net: Analyzes each frame independently.
    
    Extracts spatial features from individual frames.
    """
    
    def __init__(self, base_channels=32):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_block(1, base_channels)
        self.enc2 = self._make_block(base_channels, base_channels * 2)
        self.enc3 = self._make_block(base_channels * 2, base_channels * 4)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec1 = self._make_block(base_channels * 4, base_channels * 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec2 = self._make_block(base_channels * 2, base_channels)
    
    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) single frame
        Returns:
            features: (B, base_channels, H, W)
        """
        # Encoder
        e1 = self.enc1(x)  # (B, 32, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 64, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 128, H/4, W/4)
        
        # Decoder
        d1 = self.up1(e3)  # (B, 64, H/2, W/2)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)  # (B, 64, H/2, W/2)
        
        d2 = self.up2(d1)  # (B, 32, H, W)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)  # (B, 32, H, W)
        
        return d2


class TemporalContextNet(nn.Module):
    """
    Second U-Net: Integrates features across 3 frames.
    
    Combines spatial features with temporal context.
    """
    
    def __init__(self, base_channels=32):
        super().__init__()
        
        # Input: 3 frames × base_channels = 3*base_channels
        self.enc1 = self._make_block(3 * base_channels, base_channels * 2)
        self.enc2 = self._make_block(base_channels * 2, base_channels * 4)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec1 = self._make_block(base_channels * 4, base_channels * 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec2 = self._make_block(base_channels * 3, base_channels)
    
    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3*base_channels, H, W) concatenated 3-frame features
        Returns:
            features: (B, base_channels, H, W)
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        d1 = self.up1(e2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, x[:, :32]], dim=1)  # Skip from input
        d2 = self.dec2(d2)
        
        return d2


class DECODENet(nn.Module):
    """
    Full DECODE network: Frame Analysis + Temporal Context + Output heads.
    
    Outputs:
    - Detection probability p ∈ [0, 1]
    - Sub-pixel offsets (Δx, Δy) ∈ [-0.5, 0.5]
    - Photon count N > 0
    - Uncertainties (σx, σy, σN) > 0
    """
    
    def __init__(self, base_channels=32):
        super().__init__()
        
        self.frame_net = FrameAnalysisNet(base_channels)
        self.temporal_net = TemporalContextNet(base_channels)
        
        # Output heads
        self.prob_head = nn.Conv2d(base_channels, 1, 1)
        self.offset_head = nn.Conv2d(base_channels, 2, 1)
        self.photon_head = nn.Conv2d(base_channels, 1, 1)
        self.uncertainty_head = nn.Conv2d(base_channels, 3, 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) three consecutive frames
            
        Returns:
            dict with:
                'prob': (B, 1, H, W) detection probability
                'offset': (B, 2, H, W) sub-pixel offsets
                'photons': (B, 1, H, W) photon counts
                'uncertainty': (B, 3, H, W) uncertainties
        """
        B, T, H, W = x.shape
        assert T == 3, "DECODE requires exactly 3 frames"
        
        # Process each frame independently
        frame_features = []
        for t in range(T):
            feat = self.frame_net(x[:, t:t+1])  # (B, base_channels, H, W)
            frame_features.append(feat)
        
        # Concatenate temporal features
        temporal_input = torch.cat(frame_features, dim=1)  # (B, 3*base_channels, H, W)
        
        # Temporal integration
        features = self.temporal_net(temporal_input)  # (B, base_channels, H, W)
        
        # Output heads
        prob = torch.sigmoid(self.prob_head(features))
        offset = torch.tanh(self.offset_head(features)) * 0.5
        photons = F.softplus(self.photon_head(features))
        uncertainty = F.softplus(self.uncertainty_head(features))
        
        return {
            'prob': prob,
            'offset': offset,
            'photons': photons,
            'uncertainty': uncertainty
        }
    
    def predict(self, x, threshold=0.5):
        """
        Predict puncta coordinates from images.
        
        Args:
            x: (B, 3, H, W) three frames
            threshold: Detection threshold
            
        Returns:
            list of detections per batch:
                Each detection: (x, y, photons, σx, σy, σN)
        """
        outputs = self.forward(x)
        
        prob = outputs['prob'].cpu().numpy()
        offset = outputs['offset'].cpu().numpy()
        photons = outputs['photons'].cpu().numpy()
        uncertainty = outputs['uncertainty'].cpu().numpy()
        
        batch_detections = []
        
        for b in range(prob.shape[0]):
            # Find detections
            y_det, x_det = np.where(prob[b, 0] > threshold)
            
            detections = []
            for y, x in zip(y_det, x_det):
                # Sub-pixel position
                x_sub = x + offset[b, 0, y, x]
                y_sub = y + offset[b, 1, y, x]
                
                # Other properties
                N = photons[b, 0, y, x]
                sigma_x = uncertainty[b, 0, y, x]
                sigma_y = uncertainty[b, 1, y, x]
                sigma_N = uncertainty[b, 2, y, x]
                
                detections.append({
                    'x': x_sub,
                    'y': y_sub,
                    'photons': N,
                    'sigma_x': sigma_x,
                    'sigma_y': sigma_y,
                    'sigma_N': sigma_N,
                    'prob': prob[b, 0, y, x]
                })
            
            batch_detections.append(detections)
        
        return batch_detections


# Test the network
if __name__ == '__main__':
    import numpy as np
    
    print("Testing DECODE Network...")
    
    # Create model
    model = DECODENet(base_channels=32)
    
    # Test input (batch=2, 3 frames, 256×256)
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    outputs = model(x)
    
    print("\n✅ Forward pass successful")
    print(f"Output shapes:")
    print(f"  Probability: {outputs['prob'].shape}")
    print(f"  Offset: {outputs['offset'].shape}")
    print(f"  Photons: {outputs['photons'].shape}")
    print(f"  Uncertainty: {outputs['uncertainty'].shape}")
    
    # Test prediction
    detections = model.predict(x, threshold=0.5)
    print(f"\n✅ Prediction successful")
    print(f"Detections in batch 0: {len(detections[0])}")
    print(f"Detections in batch 1: {len(detections[1])}")
    
    # Model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model parameters: {num_params:,}")
    
    print("\n✅ All tests passed!")
