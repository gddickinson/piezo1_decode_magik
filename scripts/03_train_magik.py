#!/usr/bin/env python3
"""
Train MAGIK Graph Neural Network for Tracking

Trains the MAGIK model to link detections across frames using graph neural networks.

Usage:
    python 03_train_magik.py \\
        --config configs/magik_training.yaml \\
        --data data/synthetic \\
        --decode checkpoints/decode/best_model.pth \\
        --output checkpoints/magik
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.magik_gnn import MAGIKNet, build_tracking_graph
from piezo1_magik.models.decode_net import DECODENet


class MAGIKLoss(nn.Module):
    """Loss for training MAGIK."""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, pred_probs, edge_index, gt_tracks):
        """
        Compute loss.
        
        Args:
            pred_probs: (E,) predicted edge probabilities
            edge_index: (2, E) edge connections
            gt_tracks: Ground truth track IDs
            
        Returns:
            loss: Binary cross-entropy loss
        """
        # Create ground truth labels
        # Edge is positive if both nodes belong to same track
        src, dst = edge_index
        gt_labels = (gt_tracks[src] == gt_tracks[dst]).float()
        
        # BCE loss
        loss = self.bce(pred_probs, gt_labels)
        
        return loss


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        # Move to device
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_features = batch['edge_features'].to(device)
        gt_tracks = batch['gt_tracks'].to(device)
        
        # Forward
        optimizer.zero_grad()
        edge_probs = model(node_features, edge_index, edge_features)
        
        # Loss
        loss = criterion(edge_probs, edge_index, gt_tracks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser(description='Train MAGIK model')
    parser.add_argument('--config', type=str, required=True,
                        help='Config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory with synthetic samples')
    parser.add_argument('--decode', type=str, required=True,
                        help='Trained DECODE model (for getting detections)')
    parser.add_argument('--output', type=str, default='checkpoints/magik',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MAGIK TRAINING")
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
    model = MAGIKNet(
        node_features=config['model']['node_features'],
        edge_features=config['model']['edge_features'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers']
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Loss
    criterion = MAGIKLoss()
    
    print("\n⚠️  Note: Complete MAGIK training pipeline requires:")
    print("1. Load synthetic data with ground truth tracks")
    print("2. Run DECODE to get detections")
    print("3. Build graphs with ground truth labels")
    print("4. Train MAGIK to predict correct links")
    print("\nThis is a template script showing the structure.")
    print(f"Output will be saved to: {args.output}")
    
    print(f"\n✅ MAGIK training template ready")


if __name__ == '__main__':
    main()
