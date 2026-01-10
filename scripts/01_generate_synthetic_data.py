#!/usr/bin/env python3
"""
Generate Synthetic Training Data with Tracks

Creates realistic PIEZO1-HaloTag movies with ground truth trajectories
for training DECODE and MAGIK.

Usage:
    python 01_generate_synthetic_data.py \\
        --output data/synthetic \\
        --num_samples 5000 \\
        --num_tracks 10 \\
        --num_frames 100
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.data.synthetic_generator import generate_synthetic_movie


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of movies to generate')
    parser.add_argument('--num_tracks', type=int, default=10,
                        help='Number of tracks per movie')
    parser.add_argument('--num_frames', type=int, default=100,
                        help='Frames per movie')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Image size (H W)')
    parser.add_argument('--photons_mean', type=int, default=1000,
                        help='Mean photon count')
    parser.add_argument('--with_blinking', action='store_true',
                        help='Add blinking dynamics')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SYNTHETIC DATA GENERATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Output: {args.output}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Tracks per sample: {args.num_tracks}")
    print(f"  Frames per sample: {args.num_frames}")
    print(f"  Image size: {args.image_size}")
    print(f"  Blinking: {args.with_blinking}")
    print()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    for i in tqdm(range(args.num_samples), desc='Generating'):
        sample_dir = output_dir / f'sample_{i:05d}'
        
        generate_synthetic_movie(
            num_tracks=args.num_tracks,
            num_frames=args.num_frames,
            image_size=tuple(args.image_size),
            photons_mean=args.photons_mean,
            photons_std=200,
            with_blinking=args.with_blinking,
            output_dir=sample_dir
        )
    
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Generated {args.num_samples} samples")
    print(f"Saved to: {output_dir}")
    print(f"\nEach sample contains:")
    print(f"  - movie.tif (TIFF stack)")
    print(f"  - ground_truth_tracks.csv (track coordinates)")
    print(f"  - metadata.json (parameters)")


if __name__ == '__main__':
    main()
