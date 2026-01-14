#!/usr/bin/env python3
"""
Resume MAGIK Training from Checkpoint

Resumes interrupted MAGIK training from the last saved checkpoint.

Usage:
    python scripts/15_resume_magik_training.py \
        --checkpoint checkpoints/magik \
        --data data/synthetic_optimized \
        --decode checkpoints/decode_optimized/best_model.pth \
        --num_samples 50
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.magik_gnn import MAGIKNet
from piezo1_magik.models.decode_net import DECODENet
from piezo1_magik.data.magik_dataset import MAGIKDataset
from torch.utils.data import DataLoader, random_split


class WeightedBCELoss(nn.Module):
    """Weighted BCE loss for class imbalance."""
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        """Compute loss and metrics."""
        # BCE with logits
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=torch.tensor([self.pos_weight]).to(logits.device)
        )
        
        # Metrics
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        correct = (preds == targets).float().sum()
        total = targets.numel()
        accuracy = correct / total
        
        # Precision and recall
        true_pos = ((preds == 1) & (targets == 1)).float().sum()
        pred_pos = (preds == 1).float().sum()
        actual_pos = (targets == 1).float().sum()
        
        precision = true_pos / pred_pos if pred_pos > 0 else torch.tensor(0.0)
        recall = true_pos / actual_pos if actual_pos > 0 else torch.tensor(0.0)
        
        metrics = {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item()
        }
        
        return bce, metrics


def find_last_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for checkpoints
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    
    if len(checkpoints) == 0:
        # Check for best_model.pth
        best_path = checkpoint_dir / 'best_model.pth'
        if best_path.exists():
            return best_path, 'best_model'
        
        # Check for final_model.pth
        final_path = checkpoint_dir / 'final_model.pth'
        if final_path.exists():
            return final_path, 'final_model'
        
        return None, None
    
    # Find latest epoch checkpoint
    epoch_nums = []
    for cp in checkpoints:
        try:
            epoch = int(cp.stem.split('_')[-1])
            epoch_nums.append((epoch, cp))
        except:
            continue
    
    if len(epoch_nums) == 0:
        return None, None
    
    epoch_nums.sort(reverse=True)
    last_epoch, last_checkpoint = epoch_nums[0]
    
    return last_checkpoint, f'epoch_{last_epoch}'


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and restore training state."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get training state
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    config = checkpoint.get('config', None)
    
    print(f"✅ Loaded checkpoint from epoch {start_epoch - 1}")
    print(f"   Best val loss so far: {best_val_loss:.4f}")
    
    return start_epoch, best_val_loss, config


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_loss, 
                   config, output_dir, is_best=False):
    """Save training checkpoint."""
    output_dir = Path(output_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config
    }
    
    # Save periodic checkpoint
    checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = output_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
    
    # Clean up old checkpoints (keep last 3)
    checkpoints = sorted(output_dir.glob('checkpoint_epoch_*.pth'))
    if len(checkpoints) > 3:
        for old_cp in checkpoints[:-3]:
            old_cp.unlink()


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
        
        # Update progress
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{metrics["accuracy"]:.3f}',
            'prec': f'{metrics["precision"]:.3f}',
            'rec': f'{metrics["recall"]:.3f}'
        })
    
    # Average
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()} if num_batches > 0 else total_metrics
    
    return avg_loss, avg_metrics


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    
    total_loss = 0
    total_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            if batch['edge_index'].shape[1] == 0:
                continue
            
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            edge_labels = batch['edge_labels'].to(device)
            
            edge_logits = model(node_features, edge_index, edge_features)
            loss, metrics = criterion(edge_logits, edge_labels)
            
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()} if num_batches > 0 else total_metrics
    
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Resume MAGIK training')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint directory to resume from')
    parser.add_argument('--data', type=str, required=True,
                       help='Training data directory')
    parser.add_argument('--decode', type=str, required=True,
                       help='DECODE model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to use')
    parser.add_argument('--additional_epochs', type=int, default=None,
                       help='Additional epochs to train (default: continue to original total)')
    parser.add_argument('--gpu', type=int, default=-1)
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint)
    
    print("="*70)
    print("RESUME MAGIK TRAINING")
    print("="*70)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Data: {args.data}")
    print(f"DECODE model: {args.decode}")
    print(f"Samples: {args.num_samples}")
    print("="*70)
    
    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")
    
    # Find last checkpoint
    last_checkpoint, checkpoint_type = find_last_checkpoint(checkpoint_dir)
    
    if last_checkpoint is None:
        print("❌ No checkpoints found in directory!")
        print(f"   Looked in: {checkpoint_dir}")
        print("\nTo start fresh training, use:")
        print("   python scripts/03_train_magik.py \\")
        print(f"       --output {checkpoint_dir} \\")
        print("       ...")
        return
    
    print(f"Found checkpoint: {last_checkpoint.name} ({checkpoint_type})")
    
    # Load DECODE model
    print("\nLoading DECODE model...")
    decode_checkpoint = torch.load(args.decode, map_location=device, weights_only=False)
    decode_config = decode_checkpoint.get('config', {})
    base_channels = decode_config.get('model', {}).get('base_channels', 32)
    
    decode_model = DECODENet(base_channels=base_channels)
    decode_model.load_state_dict(decode_checkpoint['model_state_dict'])
    decode_model = decode_model.to(device)
    decode_model.eval()
    
    decode_epoch = decode_checkpoint.get('epoch', 'unknown')
    print(f"✅ DECODE loaded from epoch {decode_epoch}\n")
    
    # Create dataset
    print("Creating datasets...")
    print("(This will run DECODE on all frames to get detections...)")
    
    # Load config from checkpoint to get graph parameters
    temp_checkpoint = torch.load(last_checkpoint, map_location='cpu', weights_only=False)
    saved_config = temp_checkpoint.get('config', {})
    
    if saved_config is None:
        print("⚠️  No config found in checkpoint, using defaults")
        max_temporal_gap = 2
        max_spatial_distance = 30
    else:
        max_temporal_gap = saved_config.get('data', {}).get('max_temporal_gap', 2)
        max_spatial_distance = saved_config.get('data', {}).get('max_spatial_distance', 30)
    
    dataset = MAGIKDataset(
        args.data, decode_model, device,
        max_temporal_gap=max_temporal_gap,
        max_spatial_distance=max_spatial_distance
    )
    
    num_samples = min(args.num_samples, len(dataset))
    dataset_subset = torch.utils.data.Subset(dataset, range(num_samples))
    
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_dataset, val_dataset = random_split(
        dataset_subset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}\n")
    
    # Create model
    print("Creating MAGIK model...")
    
    if saved_config is None:
        node_features = 5
        edge_features = 4
        hidden_dim = 64
        num_layers = 3
        pos_weight = 2.0
        learning_rate = 0.001
        num_epochs = 30
    else:
        node_features = saved_config.get('model', {}).get('node_features', 5)
        edge_features = saved_config.get('model', {}).get('edge_features', 4)
        hidden_dim = saved_config.get('model', {}).get('hidden_dim', 64)
        num_layers = saved_config.get('model', {}).get('num_layers', 3)
        pos_weight = saved_config.get('training', {}).get('pos_weight', 2.0)
        learning_rate = saved_config.get('training', {}).get('learning_rate', 0.001)
        num_epochs = saved_config.get('training', {}).get('num_epochs', 30)
    
    model = MAGIKNet(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Load checkpoint
    start_epoch, best_val_loss, loaded_config = load_checkpoint(
        last_checkpoint, model, optimizer, scheduler, device
    )
    
    # Determine total epochs
    if args.additional_epochs is not None:
        total_epochs = start_epoch + args.additional_epochs
        print(f"\nTraining for {args.additional_epochs} additional epochs")
    else:
        total_epochs = num_epochs
        print(f"\nContinuing to epoch {total_epochs}")
    
    print(f"Starting from epoch {start_epoch}")
    print(f"Total epochs: {total_epochs}")
    
    # Loss function
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    
    print("\n" + "="*70)
    print("RESUMING TRAINING")
    print("="*70)
    
    # Training loop
    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch}:")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Print stats
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
        
        # Check if best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  🏆 New best model! Val loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        config = {
            'model': {
                'node_features': node_features,
                'edge_features': edge_features,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers
            },
            'training': {
                'pos_weight': pos_weight,
                'learning_rate': learning_rate,
                'num_epochs': total_epochs
            },
            'data': {
                'max_temporal_gap': max_temporal_gap,
                'max_spatial_distance': max_spatial_distance
            }
        }
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, best_val_loss,
            config, checkpoint_dir, is_best=is_best
        )
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pth'
    final_checkpoint = {
        'epoch': total_epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_loss': best_val_loss
    }
    torch.save(final_checkpoint, final_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
