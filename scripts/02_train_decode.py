#!/usr/bin/env python3
"""
Train DECODE Network for Puncta Localization

Trains the DECODE model on synthetic data with ground truth coordinates.

Usage:
    python 02_train_decode.py \\
        --config configs/decode_training.yaml \\
        --data data/synthetic \\
        --output checkpoints/decode
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.decode_net import DECODENet


class DECODELoss(nn.Module):
    """Loss function for DECODE training."""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Compute DECODE loss.
        
        Args:
            pred: Dict with 'prob', 'offset', 'photons', 'uncertainty'
            target: Dict with ground truth
            
        Returns:
            loss: Total loss
            components: Dict of loss components
        """
        # Detection loss
        det_loss = self.bce(pred['prob'], target['has_puncta'])
        
        # Only compute other losses where puncta exist
        mask = target['has_puncta'] > 0.5
        
        if mask.sum() > 0:
            # Offset loss
            offset_loss = self.mse(pred['offset'][mask], target['offset'][mask])
            
            # Photon loss
            photon_loss = self.mse(pred['photons'][mask], target['photons'][mask])
        else:
            offset_loss = torch.tensor(0.0, device=pred['prob'].device)
            photon_loss = torch.tensor(0.0, device=pred['prob'].device)
        
        # Total loss (weighted)
        total_loss = det_loss + offset_loss + 0.1 * photon_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'detection': det_loss.item(),
            'offset': offset_loss.item(),
            'photon': photon_loss.item()
        }


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        # Move to device
        images = batch['images'].to(device)  # (B, 3, H, W)
        has_puncta = batch['has_puncta'].to(device)
        offset = batch['offset'].to(device)
        photons = batch['photons'].to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        
        # Loss
        target = {
            'has_puncta': has_puncta,
            'offset': offset,
            'photons': photons
        }
        
        loss, components = criterion(outputs, target)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', leave=False):
            images = batch['images'].to(device)
            has_puncta = batch['has_puncta'].to(device)
            offset = batch['offset'].to(device)
            photons = batch['photons'].to(device)
            
            outputs = model(images)
            
            target = {
                'has_puncta': has_puncta,
                'offset': offset,
                'photons': photons
            }
            
            loss, _ = criterion(outputs, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description='Train DECODE model')
    parser.add_argument('--config', type=str, required=True,
                        help='Config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory')
    parser.add_argument('--output', type=str, default='checkpoints/decode',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DECODE TRAINING")
    print("="*70)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Device
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = DECODENet(base_channels=config['model']['base_channels'])
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Loss
    criterion = DECODELoss()
    
    # TODO: Create data loaders (implement dataset in piezo1_magik/data/dataset.py)
    print("\n⚠️  Note: Dataset implementation needed in piezo1_magik/data/dataset.py")
    print("This is a template training script.")
    print("\nTo complete:")
    print("1. Implement DECODEDataset in data/dataset.py")
    print("2. Load synthetic data with 3-frame windows")
    print("3. Run this script")
    
    print(f"\n✅ Training template ready")
    print(f"Output will be saved to: {args.output}")


if __name__ == '__main__':
    main()
