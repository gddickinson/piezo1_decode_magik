#!/usr/bin/env python3
"""
Find and Remove Corrupted Synthetic Data Files

Scans through synthetic data directory and identifies corrupted TIFF files.

Usage:
    python scripts/10_fix_corrupted_data.py \
        --data data/synthetic_optimized \
        --remove
"""

import argparse
import tifffile
from pathlib import Path
from tqdm import tqdm
import shutil


def check_sample(sample_dir):
    """Check if a sample is valid."""
    movie_path = sample_dir / 'movie.tif'
    gt_path = sample_dir / 'ground_truth_tracks.csv'
    
    # Check if files exist
    if not movie_path.exists():
        return False, "Missing movie.tif"
    
    if not gt_path.exists():
        return False, "Missing ground_truth_tracks.csv"
    
    # Try to read TIFF
    try:
        movie = tifffile.imread(movie_path)
        
        # Check shape
        if len(movie.shape) != 3:
            return False, f"Invalid shape: {movie.shape} (expected 3D)"
        
        # Check size
        if movie.size == 0:
            return False, "Empty movie"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"TIFF error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Find corrupted data files')
    parser.add_argument('--data', type=str, required=True,
                       help='Synthetic data directory')
    parser.add_argument('--remove', action='store_true',
                       help='Remove corrupted samples')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    
    print("="*70)
    print("CHECKING SYNTHETIC DATA")
    print("="*70)
    print(f"Directory: {data_dir}")
    print("="*70)
    
    # Find all samples
    samples = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"\nFound {len(samples)} samples")
    print("Checking each sample...\n")
    
    corrupted = []
    
    for sample_dir in tqdm(samples, desc='Checking'):
        is_valid, message = check_sample(sample_dir)
        
        if not is_valid:
            corrupted.append((sample_dir, message))
    
    # Report results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nValid samples: {len(samples) - len(corrupted)}")
    print(f"Corrupted samples: {len(corrupted)}")
    
    if len(corrupted) > 0:
        print("\nCorrupted samples:")
        for sample_dir, message in corrupted:
            print(f"  {sample_dir.name}: {message}")
        
        if args.remove:
            print(f"\nRemoving {len(corrupted)} corrupted samples...")
            
            for sample_dir, _ in corrupted:
                shutil.rmtree(sample_dir)
                print(f"  Removed: {sample_dir.name}")
            
            print("\n✅ Corrupted samples removed")
            print(f"Remaining samples: {len(samples) - len(corrupted)}")
        else:
            print("\nTo remove corrupted samples, run with --remove flag:")
            print(f"  python {Path(__file__).name} --data {args.data} --remove")
    else:
        print("\n✅ All samples are valid!")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
