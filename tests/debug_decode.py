#!/usr/bin/env python3
"""
Debug DECODE Model - Check What It's Actually Outputting

Usage:
    python scripts/debug_decode.py \
        --model checkpoints/decode_test/best_model.pth \
        --data data/synthetic_test
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.decode_net import DECODENet
from piezo1_magik.data.decode_dataset import DECODEDataset


def main():
    parser = argparse.ArgumentParser(description='Debug DECODE model')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    
    args = parser.parse_args()
    
    print("="*70)
    print("DECODE MODEL DEBUG")
    print("="*70)
    
    # Device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")
    
    # Load model
    print(f"Loading model: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {})
    base_channels = config.get('model', {}).get('base_channels', 32)
    
    model = DECODENet(base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}\n")
    
    # Load dataset
    print(f"Loading data: {args.data}")
    dataset = DECODEDataset(args.data)
    print(f"✅ Found {len(dataset)} samples\n")
    
    # Test on first sample
    print("="*70)
    print("TESTING ON FIRST SAMPLE")
    print("="*70)
    
    sample = dataset[0]
    
    images = sample['images']  # Already a tensor (3, H, W)
    has_puncta = sample['has_puncta'].numpy()[0]  # (H, W)
    
    print(f"\nInput:")
    print(f"  Images shape: {images.shape}")
    print(f"  Images range: {images.min():.4f} - {images.max():.4f}")
    print(f"  Ground truth puncta: {(has_puncta > 0.5).sum()} pixels")
    
    # Run inference
    with torch.no_grad():
        images_batch = images.unsqueeze(0).to(device)  # (1, 3, H, W)
        outputs = model(images_batch)
    
    # Get outputs
    prob = outputs['prob'][0, 0].cpu().numpy()  # (H, W)
    offset = outputs['offset'][0].cpu().numpy()  # (2, H, W)
    photons = outputs['photons'][0, 0].cpu().numpy()  # (H, W)
    
    print(f"\nModel Outputs:")
    print(f"  Probability map shape: {prob.shape}")
    print(f"  Probability range: {prob.min():.6f} - {prob.max():.6f}")
    print(f"  Probability mean: {prob.mean():.6f}")
    print(f"  Probability std: {prob.std():.6f}")
    print(f"  Pixels > 0.5: {(prob > 0.5).sum()}")
    print(f"  Pixels > 0.3: {(prob > 0.3).sum()}")
    print(f"  Pixels > 0.1: {(prob > 0.1).sum()}")
    print(f"  Pixels > 0.01: {(prob > 0.01).sum()}")
    
    print(f"\n  Offset range: ({offset[0].min():.4f}, {offset[0].max():.4f}), ({offset[1].min():.4f}, {offset[1].max():.4f})")
    print(f"  Photons range: {photons.min():.1f} - {photons.max():.1f}")
    
    # Show where ground truth is
    gt_y, gt_x = np.where(has_puncta > 0.5)
    if len(gt_x) > 0:
        print(f"\nGround Truth Locations:")
        print(f"  Found {len(gt_x)} puncta")
        print(f"  Example locations: {list(zip(gt_x[:3], gt_y[:3]))}")
        
        # Check probabilities at GT locations
        gt_probs = prob[gt_y, gt_x]
        print(f"\n  Probabilities at GT locations:")
        print(f"    Min: {gt_probs.min():.6f}")
        print(f"    Max: {gt_probs.max():.6f}")
        print(f"    Mean: {gt_probs.mean():.6f}")
        print(f"    Median: {np.median(gt_probs):.6f}")
    
    # Visualize
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATION")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Input image (center frame)
    axes[0, 0].imshow(images[1].cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Input (Center Frame)', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth
    axes[0, 1].imshow(images[1].cpu().numpy(), cmap='gray', alpha=0.5)
    axes[0, 1].scatter(gt_x, gt_y, c='red', s=100, marker='o', 
                      edgecolors='white', linewidths=2, label='GT')
    axes[0, 1].set_title(f'Ground Truth ({len(gt_x)} puncta)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].axis('off')
    
    # Predicted probability map
    im = axes[0, 2].imshow(prob, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Predicted Probability\n(max={prob.max():.4f})', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Detections at different thresholds
    for idx, thresh in enumerate([0.5, 0.3, 0.1]):
        ax = axes[1, idx]
        
        det_mask = prob > thresh
        det_y, det_x = np.where(det_mask)
        
        ax.imshow(images[1].cpu().numpy(), cmap='gray', alpha=0.5)
        ax.scatter(gt_x, gt_y, c='red', s=100, marker='o', 
                  edgecolors='white', linewidths=2, label='GT', alpha=0.5)
        ax.scatter(det_x, det_y, c='cyan', s=50, marker='x', 
                  linewidths=2, label=f'Pred (thresh={thresh})')
        ax.set_title(f'Threshold={thresh}\n({len(det_x)} detections)', fontweight='bold')
        ax.legend()
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_output.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: debug_output.png")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    max_prob = prob.max()
    
    if max_prob < 0.1:
        print("\n⚠️  PROBLEM: Maximum probability is very low!")
        print("   The model is not confident about any detections.")
        print("\n   Possible causes:")
        print("   1. Model didn't train properly")
        print("   2. Data normalization mismatch")
        print("   3. Model overfitted to training data")
        print("\n   Try:")
        print("   - Check training loss (should be < 0.1)")
        print("   - Visualize training data to ensure it looks correct")
        print("   - Train for more epochs")
    
    elif max_prob < 0.5:
        print(f"\n⚠️  Model probabilities are low (max={max_prob:.4f})")
        print(f"   But there are {(prob > 0.1).sum()} pixels above 0.1")
        print("\n   Use lower threshold for evaluation:")
        print(f"   python scripts/05_evaluate_decode.py \\")
        print(f"       --model {args.model} \\")
        print(f"       --data {args.data} \\")
        print(f"       --threshold 0.1")
    
    else:
        print(f"\n✅ Model probabilities look reasonable (max={max_prob:.4f})")
        print(f"   Found {(prob > 0.5).sum()} detections at threshold 0.5")
    
    # Check if probabilities at GT locations are high
    if len(gt_probs) > 0:
        if gt_probs.mean() < 0.1:
            print("\n⚠️  WARNING: Low probabilities at ground truth locations!")
            print("   The model is not detecting real puncta.")
            print("   This suggests the model didn't learn properly.")
        elif gt_probs.mean() < 0.5:
            print(f"\n⚠️  Moderate probabilities at GT locations (mean={gt_probs.mean():.4f})")
            print("   Use threshold ~0.2-0.3 for evaluation")
        else:
            print(f"\n✅ Good probabilities at GT locations (mean={gt_probs.mean():.4f})")


if __name__ == '__main__':
    main()
