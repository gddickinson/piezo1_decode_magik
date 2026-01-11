#!/usr/bin/env python3
"""
Train DECODE Network for Puncta Localization

Trains the DECODE model on synthetic data with ground truth coordinates.

Usage:
    python 02_train_decode.py \
        --config configs/decode_training.yaml \
        --data data/synthetic \
        --output checkpoints/decode \
        --gpu 0
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

# Try to import TensorBoard, but make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("⚠️  TensorBoard not available, logging disabled")

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.decode_net import DECODENet
from piezo1_magik.data.decode_dataset import create_dataloaders


class DECODELoss(nn.Module):
    """Loss function for DECODE training."""
    
    def __init__(self, weights={'detection': 1.0, 'offset': 1.0, 'photon': 0.01}):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.weights = weights
    
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
        # Detection loss (all pixels)
        det_loss = self.bce(pred['prob'], target['has_puncta'])
        
        # Only compute other losses where puncta exist
        mask = target['has_puncta'] > 0.5  # (B, 1, H, W)
        mask = mask.squeeze(1)  # (B, H, W) - remove channel dimension
        
        if mask.sum() > 0:
            # Offset loss (only where puncta exist)
            # pred['offset'] is (B, 2, H, W), permute to (B, H, W, 2)
            pred_offset_permuted = pred['offset'].permute(0, 2, 3, 1)  # (B, H, W, 2)
            target_offset_permuted = target['offset'].permute(0, 2, 3, 1)  # (B, H, W, 2)
            
            # Apply mask - now both are compatible
            pred_offset_masked = pred_offset_permuted[mask]  # (N, 2)
            target_offset_masked = target_offset_permuted[mask]  # (N, 2)
            offset_loss = self.mse(pred_offset_masked, target_offset_masked)
            
            # Photon loss (log-scale to prevent explosion)
            # Photons are in range ~100-2000, so log brings them to ~4.6-7.6
            pred_photons_masked = pred['photons'][mask.unsqueeze(1)]
            target_photons_masked = target['photons'][mask.unsqueeze(1)]
            
            # Use log scale to prevent huge losses
            pred_log = torch.log(pred_photons_masked + 1)
            target_log = torch.log(target_photons_masked + 1)
            photon_loss = self.mse(pred_log, target_log)
        else:
            offset_loss = torch.tensor(0.0, device=pred['prob'].device)
            photon_loss = torch.tensor(0.0, device=pred['prob'].device)
        
        # Total loss (weighted)
        total_loss = (self.weights['detection'] * det_loss + 
                     self.weights['offset'] * offset_loss + 
                     self.weights['photon'] * photon_loss)
        
        return total_loss, {
            'total': total_loss.item(),
            'detection': det_loss.item(),
            'offset': offset_loss.item(),
            'photon': photon_loss.item()
        }


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    
    epoch_loss = 0
    epoch_det_loss = 0
    epoch_offset_loss = 0
    epoch_photon_loss = 0
    
    progress = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress):
        # Move to device
        images = batch['images'].to(device)
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
        
        # Accumulate
        epoch_loss += loss.item()
        epoch_det_loss += components['detection']
        epoch_offset_loss += components['offset']
        epoch_photon_loss += components['photon']
        
        # Update progress
        progress.set_postfix({
            'loss': f"{loss.item():.4f}",
            'det': f"{components['detection']:.4f}",
            'off': f"{components['offset']:.4f}"
        })
    
    # Average losses
    n = len(train_loader)
    avg_losses = {
        'total': epoch_loss / n,
        'detection': epoch_det_loss / n,
        'offset': epoch_offset_loss / n,
        'photon': epoch_photon_loss / n
    }
    
    # Log to tensorboard
    if writer:
        for name, value in avg_losses.items():
            writer.add_scalar(f'train/{name}', value, epoch)
    
    return avg_losses


def validate(model, val_loader, criterion, device, epoch, writer=None):
    """Validate model."""
    model.eval()
    
    epoch_loss = 0
    epoch_det_loss = 0
    epoch_offset_loss = 0
    epoch_photon_loss = 0
    
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
            
            loss, components = criterion(outputs, target)
            
            epoch_loss += loss.item()
            epoch_det_loss += components['detection']
            epoch_offset_loss += components['offset']
            epoch_photon_loss += components['photon']
    
    # Average losses
    n = len(val_loader)
    avg_losses = {
        'total': epoch_loss / n,
        'detection': epoch_det_loss / n,
        'offset': epoch_offset_loss / n,
        'photon': epoch_photon_loss / n
    }
    
    # Log to tensorboard
    if writer:
        for name, value in avg_losses.items():
            writer.add_scalar(f'val/{name}', value, epoch)
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser(description='Train DECODE model')
    parser.add_argument('--config', type=str, required=True,
                        help='Config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory with synthetic samples')
    parser.add_argument('--output', type=str, default='checkpoints/decode',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DECODE TRAINING")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Convert scientific notation strings to floats
    if isinstance(config['training']['learning_rate'], str):
        config['training']['learning_rate'] = float(config['training']['learning_rate'])
    if 'min_lr' in config['training'] and isinstance(config['training']['min_lr'], str):
        config['training']['min_lr'] = float(config['training']['min_lr'])
    
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    
    # Device
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"  Device: {device}")
    
    # Create data loaders
    print(f"\nCreating dataloaders...")
    
    # Fix for macOS: num_workers can cause "too many open files" error
    num_workers = config['data']['num_workers']
    if device == 'mps':
        num_workers = 0
        print(f"  ⚠️  Setting num_workers=0 for MPS (macOS compatibility)")
    
    train_loader, val_loader = create_dataloaders(
        args.data,
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        num_workers=num_workers,
        device=device  # Pass device to avoid pin_memory warning on MPS
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating DECODE model...")
    model = DECODENet(base_channels=config['model']['base_channels'])
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training'].get('min_lr', 1e-6)
    )
    
    # Loss
    criterion = DECODELoss()
    
    # TensorBoard (optional)
    if HAS_TENSORBOARD:
        try:
            writer = SummaryWriter(output_dir / 'logs')
        except Exception as e:
            print(f"⚠️  Could not initialize TensorBoard: {e}")
            writer = None
    else:
        writer = None
    
    # Resume if requested
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  Resuming from epoch {start_epoch}")
    
    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    try:
        for epoch in range(start_epoch, config['training']['num_epochs']):
            # Train
            train_losses = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, writer
            )
            
            # Validate
            val_losses = validate(
                model, val_loader, criterion, device, epoch, writer
            )
            
            # Step scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(det={train_losses['detection']:.4f}, "
                  f"off={train_losses['offset']:.4f}, "
                  f"phot={train_losses['photon']:.4f})")
            print(f"  Val Loss:   {val_losses['total']:.4f} "
                  f"(det={val_losses['detection']:.4f}, "
                  f"off={val_losses['offset']:.4f}, "
                  f"phot={val_losses['photon']:.4f})")
            print(f"  LR: {current_lr:.2e}")
            
            # Save checkpoint
            is_best = val_losses['total'] < best_val_loss
            if is_best:
                best_val_loss = val_losses['total']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'config': config
            }
            
            # Save latest
            torch.save(checkpoint, output_dir / 'latest.pth')
            
            # Save best
            if is_best:
                torch.save(checkpoint, output_dir / 'best_model.pth')
                print(f"  🏆 New best model! Val loss: {best_val_loss:.4f}")
            
            # Save periodic
            if (epoch + 1) % 10 == 0:
                torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {output_dir / 'best_model.pth'}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        interrupted_ckpt = output_dir / f'interrupted_epoch_{epoch}.pth'
        torch.save(checkpoint, interrupted_ckpt)
        print(f"✅ Checkpoint saved: {interrupted_ckpt.name}")
        print(f"\n📝 To resume training, run:")
        print(f"   python scripts/02_train_decode.py \\")
        print(f"       --config {args.config} \\")
        print(f"       --data {args.data} \\")
        print(f"       --output {args.output} \\")
        print(f"       --resume {interrupted_ckpt}")
        print(f"\n   Or use: python scripts/02_resume_training.py \\")
        print(f"       --checkpoint {interrupted_ckpt} \\")
        print(f"       --config {args.config} \\")
        print(f"       --data {args.data} \\")
        print(f"       --output {args.output}")
    
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print(f"\n💾 Latest checkpoint saved at: {output_dir / 'latest.pth'}")
        print(f"\n📝 To resume training, run:")
        print(f"   python scripts/02_train_decode.py \\")
        print(f"       --resume {output_dir / 'latest.pth'} \\")
        print(f"       --config {args.config} \\")
        print(f"       --data {args.data} \\")
        print(f"       --output {args.output}")
        raise
    
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
