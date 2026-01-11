#!/usr/bin/env python3
"""
Check Training Data Quality

Verifies that synthetic data was generated correctly and matches expectations.

Usage:
    python scripts/check_training_data.py --data data/synthetic_test
"""

import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Check training data quality')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    
    print("="*70)
    print("TRAINING DATA QUALITY CHECK")
    print("="*70)
    
    # Find samples
    samples = sorted(list(data_dir.glob('sample_*')))
    
    if len(samples) == 0:
        print(f"❌ No samples found in {data_dir}")
        return
    
    print(f"\nFound {len(samples)} samples")
    
    # Check first sample
    sample_dir = samples[0]
    print(f"\nChecking: {sample_dir.name}")
    
    # Load movie
    movie_path = sample_dir / 'movie.tif'
    if not movie_path.exists():
        print(f"❌ Movie not found: {movie_path}")
        return
    
    movie = tifffile.imread(movie_path)
    print(f"\n✅ Movie loaded:")
    print(f"   Shape: {movie.shape}")
    print(f"   Dtype: {movie.dtype}")
    print(f"   Range: {movie.min()} - {movie.max()}")
    print(f"   Mean: {movie.mean():.1f}")
    print(f"   Std: {movie.std():.1f}")
    
    # Load ground truth
    gt_path = sample_dir / 'ground_truth_tracks.csv'
    if gt_path.exists():
        gt_df = pd.read_csv(gt_path)
        print(f"\n✅ Ground truth loaded:")
        print(f"   Total detections: {len(gt_df)}")
        print(f"   Unique tracks: {gt_df['track_id'].nunique()}")
        print(f"   Frames: {gt_df['frame'].min()} - {gt_df['frame'].max()}")
        print(f"   Photon range: {gt_df['photons'].min():.0f} - {gt_df['photons'].max():.0f}")
        print(f"   Photon mean: {gt_df['photons'].mean():.0f}")
    
    # Check what it looks like when normalized
    movie_normalized = movie.astype(np.float32) / 65535.0
    print(f"\n📊 After normalization (/ 65535):")
    print(f"   Range: {movie_normalized.min():.6f} - {movie_normalized.max():.6f}")
    print(f"   Mean: {movie_normalized.mean():.6f}")
    print(f"   Std: {movie_normalized.std():.6f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original movie (first frame)
    frame0 = movie[0]
    axes[0, 0].imshow(frame0, cmap='gray', vmin=np.percentile(frame0, 1),
                     vmax=np.percentile(frame0, 99))
    axes[0, 0].set_title(f'Frame 0 (Raw)\nRange: {frame0.min()}-{frame0.max()}',
                        fontweight='bold')
    axes[0, 0].axis('off')
    
    # Middle frame
    frame_mid = movie[len(movie)//2]
    axes[0, 1].imshow(frame_mid, cmap='gray', vmin=np.percentile(frame_mid, 1),
                     vmax=np.percentile(frame_mid, 99))
    axes[0, 1].set_title(f'Frame {len(movie)//2} (Raw)\nRange: {frame_mid.min()}-{frame_mid.max()}',
                        fontweight='bold')
    axes[0, 1].axis('off')
    
    # Histogram of pixel values
    axes[0, 2].hist(frame_mid.flatten(), bins=100, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Pixel Value (uint16)', fontsize=10)
    axes[0, 2].set_ylabel('Count', fontsize=10)
    axes[0, 2].set_title('Pixel Value Distribution', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Normalized versions
    frame0_norm = frame0.astype(np.float32) / 65535.0
    axes[1, 0].imshow(frame0_norm, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Frame 0 (Normalized)\nRange: {frame0_norm.min():.4f}-{frame0_norm.max():.4f}',
                        fontweight='bold')
    axes[1, 0].axis('off')
    
    frame_mid_norm = frame_mid.astype(np.float32) / 65535.0
    axes[1, 1].imshow(frame_mid_norm, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Frame {len(movie)//2} (Normalized)\nRange: {frame_mid_norm.min():.4f}-{frame_mid_norm.max():.4f}',
                        fontweight='bold')
    axes[1, 1].axis('off')
    
    # Histogram of normalized values
    axes[1, 2].hist(frame_mid_norm.flatten(), bins=100, edgecolor='black', alpha=0.7)
    axes[1, 2].set_xlabel('Normalized Value', fontsize=10)
    axes[1, 2].set_ylabel('Count', fontsize=10)
    axes[1, 2].set_title('Normalized Distribution', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig('data_quality_check.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: data_quality_check.png")
    
    # Diagnosis
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")
    
    # Check if data looks reasonable
    if movie.max() < 200:
        print("\n❌ PROBLEM: Movie is extremely dark!")
        print(f"   Maximum pixel value is only {movie.max()}")
        print(f"   Expected range: ~100-2000 for realistic data")
        print("\n   This means synthetic data generation failed.")
        print("\n   Solutions:")
        print("   1. Check synthetic_generator.py - photon counts might be too low")
        print("   2. Regenerate synthetic data with higher photon counts")
        print("   3. Check add_noise() function - baseline might be wrong")
    
    elif movie_normalized.max() < 0.01:
        print("\n❌ PROBLEM: Normalized movie is too dark!")
        print(f"   Maximum after normalization: {movie_normalized.max():.6f}")
        print("   Expected: 0.01-0.05 for realistic puncta")
        print("\n   This will cause training to fail.")
    
    elif movie.mean() < 80:
        print("\n⚠️  WARNING: Low baseline")
        print(f"   Mean pixel value: {movie.mean():.1f}")
        print("   Expected: ~100-120 for typical camera baseline")
    
    else:
        print("\n✅ Data looks reasonable!")
        print(f"   Range: {movie.min()}-{movie.max()}")
        print(f"   This should work for training.")
        print("\n   If model still doesn't work, the issue is elsewhere:")
        print("   - Check model architecture")
        print("   - Check loss function")
        print("   - Try longer training")
    
    # Check one more thing - are there actually puncta visible?
    print(f"\n{'='*70}")
    print("VISUAL INSPECTION")
    print(f"{'='*70}")
    
    # Find a frame with puncta
    if gt_path.exists():
        # Get frame with most puncta
        frame_counts = gt_df.groupby('frame').size()
        best_frame = frame_counts.idxmax()
        frame_puncta = gt_df[gt_df['frame'] == best_frame]
        
        print(f"\nFrame {best_frame} has {len(frame_puncta)} puncta")
        print(f"Puncta locations:")
        for _, row in frame_puncta.iterrows():
            x, y = int(row['x']), int(row['y'])
            if 0 <= y < movie.shape[1] and 0 <= x < movie.shape[2]:
                pixel_val = movie[best_frame, y, x]
                print(f"   ({x}, {y}): pixel value = {pixel_val}")
        
        # Show this frame
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        frame_show = movie[best_frame].astype(np.float32) / 65535.0
        ax.imshow(frame_show, cmap='gray', vmin=0, vmax=frame_show.max())
        
        # Mark puncta
        ax.scatter(frame_puncta['x'], frame_puncta['y'], 
                  c='red', s=200, marker='o', 
                  edgecolors='white', linewidths=2, alpha=0.7)
        
        ax.set_title(f'Frame {best_frame} with {len(frame_puncta)} Puncta\n' +
                    f'Are they visible?', fontweight='bold', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('puncta_visibility.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: puncta_visibility.png")
        print("\n   👁️  LOOK AT THIS IMAGE!")
        print("   Can you see bright spots at the red circles?")
        print("   - YES → Data is good, model architecture issue")
        print("   - NO → Data generation failed, regenerate data")


if __name__ == '__main__':
    main()
