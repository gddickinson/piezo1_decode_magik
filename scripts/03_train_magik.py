#!/usr/bin/env python3
"""
Train MAGIK Graph Neural Network for Tracking

Trains the MAGIK model to link detections across frames using graph neural networks.

Usage:
    python scripts/03_train_magik.py \
        --config configs/magik_training.yaml \
        --data data/synthetic_optimized \
        --decode checkpoints/decode_optimized/best_model.pth \
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
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.magik_gnn import MAGIKNet
from piezo1_magik.models.decode_net import DECODENet
from piezo1_magik.data.magik_dataset import MAGIKDataset, collate_graphs


class MAGIKLoss(nn.Module):
    """Loss for training MAGIK with class balancing."""
    
    def __init__(self, pos_weight=2.0):
        super().__init__()
        # Weighted BCE to handle class imbalance
        # (typically more negative edges than positive)
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    
    def forward(self, pred_logits, edge_labels):
        """
        Compute loss.
        
        Args:
            pred_logits: (E,) predicted edge logits (before sigmoid)
            edge_labels: (E,) ground truth labels (0 or 1)
            
        Returns:
            loss: Binary cross-entropy loss
            metrics: Dict with accuracy, precision, recall
        """
        # BCE loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits, edge_labels, 
            pos_weight=torch.tensor([self.pos_weight]).to(pred_logits.device)
        )
        
        # Compute metrics
        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_logits)
            pred_labels = (pred_probs > 0.5).float()
            
            correct = (pred_labels == edge_labels).sum().item()
            total = edge_labels.numel()
            accuracy = correct / total if total > 0 else 0
            
            # Precision and recall
            tp = ((pred_labels == 1) & (edge_labels == 1)).sum().item()
            fp = ((pred_labels == 1) & (edge_labels == 0)).sum().item()
            fn = ((pred_labels == 0) & (edge_labels == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        
        return loss, metrics


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0}
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # Skip empty batches
        if batch['edge_index'].shape[1] == 0:
            continue
        
        # Move to device
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_features = batch['edge_features'].to(device)
        edge_labels = batch['edge_labels'].to(device)
        
        # Forward
        optimizer.zero_grad()
        edge_logits = model(node_features, edge_index, edge_features)
        
        # Loss
        loss, metrics = criterion(edge_logits, edge_labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{metrics["accuracy"]:.3f}',
            'prec': f'{metrics["precision"]:.3f}',
            'rec': f'{metrics["recall"]:.3f}'
        })
    
    if num_batches == 0:
        return 0, {'accuracy': 0, 'precision': 0, 'recall': 0}
    
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            # Skip empty batches
            if batch['edge_index'].shape[1] == 0:
                continue
            
            # Move to device
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            edge_labels = batch['edge_labels'].to(device)
            
            # Forward
            edge_logits = model(node_features, edge_index, edge_features)
            
            # Loss
            loss, metrics = criterion(edge_logits, edge_labels)
            
            # Accumulate
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_batches += 1
    
    if num_batches == 0:
        return 0, {'accuracy': 0, 'precision': 0, 'recall': 0}
    
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


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
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to use for training')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MAGIK TRAINING")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"DECODE model: {args.decode}")
    print(f"Output: {args.output}")
    print(f"Samples: {args.num_samples}")
    print("="*70)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")
    
    # Load DECODE model
    print("Loading DECODE model...")
    decode_checkpoint = torch.load(args.decode, map_location=device, weights_only=False)
    decode_config = decode_checkpoint.get('config', {})
    base_channels = decode_config.get('model', {}).get('base_channels', 32)
    
    decode_model = DECODENet(base_channels=base_channels)
    decode_model.load_state_dict(decode_checkpoint['model_state_dict'])
    decode_model = decode_model.to(device)
    decode_model.eval()
    print(f"✅ DECODE loaded from epoch {decode_checkpoint['epoch']}\n")
    
    # Create datasets
    print("Creating datasets...")
    print("(This will run DECODE on all frames to get detections...)")
    
    full_dataset = MAGIKDataset(
        args.data,
        decode_model,
        device=device,
        max_temporal_gap=config['data']['max_temporal_gap'],
        max_spatial_distance=config['data']['max_spatial_distance']
    )
    
    # Limit to num_samples
    num_samples = min(args.num_samples, len(full_dataset))
    indices = list(range(num_samples))
    
    # Split train/val
    train_split = config['data'].get('train_split', 0.8)
    num_train = int(num_samples * train_split)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_samples]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}\n")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0  # Set to 0 for MPS compatibility
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=0
    )
    
    # Create MAGIK model
    print("Creating MAGIK model...")
    model = MAGIKNet(
        node_features=config['model']['node_features'],
        edge_features=config['model']['edge_features'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers']
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ MAGIK model created")
    print(f"   Parameters: {total_params:,}\n")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Loss
    criterion = MAGIKLoss(pos_weight=config['training'].get('pos_weight', 2.0))
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch}:")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # Step scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} " +
              f"(acc={train_metrics['accuracy']:.3f}, " +
              f"prec={train_metrics['precision']:.3f}, " +
              f"rec={train_metrics['recall']:.3f})")
        print(f"  Val Loss:   {val_loss:.4f} " +
              f"(acc={val_metrics['accuracy']:.3f}, " +
              f"prec={val_metrics['precision']:.3f}, " +
              f"rec={val_metrics['recall']:.3f})")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / 'latest.pth')
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  🏆 New best model! Val loss: {val_loss:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
