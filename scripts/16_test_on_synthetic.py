#!/usr/bin/env python3
"""
Test Pipeline on Held-Out Synthetic Samples

Tests DECODE+MAGIK on synthetic samples that weren't used in training.
Compares results to ground truth.

Usage:
    python scripts/16_test_on_synthetic.py \
        --data data/synthetic_optimized \
        --decode checkpoints/decode_optimized/best_model.pth \
        --magik checkpoints/magik/best_model.pth \
        --output results/test_synthetic \
        --start_sample 150 \
        --num_samples 10
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import json


def run_pipeline_on_sample(sample_dir, decode_path, magik_path, output_dir, 
                           threshold=0.3, use_gap_filling=True):
    """Run pipeline on one sample."""
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    
    movie_path = sample_dir / 'movie.tif'
    sample_output = output_dir / sample_dir.name
    
    # Build command
    cmd = [
        'python', 'scripts/09_run_complete_pipeline.py',
        '--input', str(movie_path),
        '--decode', str(decode_path),
        '--magik', str(magik_path),
        '--output', str(sample_output),
        '--threshold', str(threshold),
        '--detection_threshold', '0.5'
    ]
    
    if use_gap_filling:
        cmd.extend(['--use_gap_filling', '--max_gap', '40', '--max_distance', '100'])
    
    # Run
    print(f"\n{'='*70}")
    print(f"Processing: {sample_dir.name}")
    print(f"{'='*70}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")
        return False


def compare_to_ground_truth(sample_dir, results_dir):
    """Compare results to ground truth."""
    sample_dir = Path(sample_dir)
    results_dir = Path(results_dir)
    
    # Load ground truth
    gt_path = sample_dir / 'ground_truth_tracks.csv'
    gt = pd.read_csv(gt_path)
    
    # Load results
    tracks_path = results_dir / 'tracks.csv'
    tracks = pd.read_csv(tracks_path)
    
    summary_path = results_dir / 'track_summary.csv'
    summary = pd.read_csv(summary_path)
    
    # Analyze
    metrics = {}
    
    # Detection metrics
    gt_detections = len(gt)
    pred_detections = len(tracks[tracks['track_id'] >= 0])
    metrics['gt_detections'] = gt_detections
    metrics['pred_detections'] = pred_detections
    
    # Match detections (within 3 pixels)
    matches = 0
    for frame in gt['frame'].unique():
        gt_frame = gt[gt['frame'] == frame]
        pred_frame = tracks[tracks['frame'] == frame]
        
        for _, gt_det in gt_frame.iterrows():
            if len(pred_frame) == 0:
                continue
            
            distances = np.sqrt((pred_frame['x'] - gt_det['x'])**2 + 
                              (pred_frame['y'] - gt_det['y'])**2)
            
            if distances.min() < 3.0:
                matches += 1
    
    metrics['matches'] = matches
    metrics['precision'] = matches / pred_detections if pred_detections > 0 else 0
    metrics['recall'] = matches / gt_detections if gt_detections > 0 else 0
    metrics['f1'] = (2 * metrics['precision'] * metrics['recall'] / 
                    (metrics['precision'] + metrics['recall']) 
                    if (metrics['precision'] + metrics['recall']) > 0 else 0)
    
    # Track metrics
    gt_tracks = gt.groupby('track_id')
    gt_track_lengths = gt_tracks.size()
    
    metrics['gt_num_tracks'] = len(gt_track_lengths)
    metrics['pred_num_tracks'] = len(summary)
    metrics['gt_mean_length'] = gt_track_lengths.mean()
    metrics['pred_mean_length'] = summary['length'].mean()
    metrics['fragmentation'] = metrics['pred_num_tracks'] / metrics['gt_num_tracks']
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Test pipeline on held-out synthetic samples'
    )
    parser.add_argument('--data', type=str, required=True,
                       help='Synthetic data directory')
    parser.add_argument('--decode', type=str, required=True,
                       help='DECODE model')
    parser.add_argument('--magik', type=str, required=True,
                       help='MAGIK model')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--start_sample', type=int, default=150,
                       help='First sample index to test')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to test')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='MAGIK threshold')
    parser.add_argument('--use_gap_filling', action='store_true',
                       default=True,
                       help='Use gap-filling')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TEST ON HELD-OUT SYNTHETIC SAMPLES")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"DECODE: {args.decode}")
    print(f"MAGIK: {args.magik}")
    print(f"Testing samples: {args.start_sample} to "
          f"{args.start_sample + args.num_samples - 1}")
    print(f"Gap-filling: {args.use_gap_filling}")
    print("="*70)
    
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sample
    all_metrics = []
    
    for i in range(args.start_sample, args.start_sample + args.num_samples):
        sample_dir = data_dir / f'sample_{i:05d}'  # Use 5 digits, not 3
        
        if not sample_dir.exists():
            print(f"⚠️  Sample {i} not found, skipping...")
            continue
        
        # Run pipeline
        success = run_pipeline_on_sample(
            sample_dir, args.decode, args.magik, output_dir,
            args.threshold, args.use_gap_filling
        )
        
        if not success:
            continue
        
        # Compare to GT
        sample_output = output_dir / sample_dir.name
        metrics = compare_to_ground_truth(sample_dir, sample_output)
        metrics['sample'] = sample_dir.name
        
        all_metrics.append(metrics)
        
        # Print sample results
        print(f"\n📊 {sample_dir.name} Results:")
        print(f"   Detection - Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
        print(f"   Tracking - Fragmentation: {metrics['fragmentation']:.1f}×, "
              f"Mean length: {metrics['pred_mean_length']:.1f}")
    
    # Save results
    if len(all_metrics) > 0:
        results_df = pd.DataFrame(all_metrics)
        results_path = output_dir / 'evaluation_results.csv'
        results_df.to_csv(results_path, index=False)
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY ACROSS ALL SAMPLES")
        print("="*70)
        
        print("\n📊 Detection Performance:")
        print(f"   Mean Precision: {results_df['precision'].mean():.3f} "
              f"(±{results_df['precision'].std():.3f})")
        print(f"   Mean Recall:    {results_df['recall'].mean():.3f} "
              f"(±{results_df['recall'].std():.3f})")
        print(f"   Mean F1:        {results_df['f1'].mean():.3f} "
              f"(±{results_df['f1'].std():.3f})")
        
        print("\n📊 Tracking Performance:")
        print(f"   Mean Fragmentation: {results_df['fragmentation'].mean():.1f}× "
              f"(±{results_df['fragmentation'].std():.1f})")
        print(f"   Mean Track Length:  {results_df['pred_mean_length'].mean():.1f} "
              f"(±{results_df['pred_mean_length'].std():.1f})")
        print(f"   GT Track Length:    {results_df['gt_mean_length'].mean():.1f}")
        
        print(f"\n✅ Results saved: {results_path}")
        
        # Expected performance
        print("\n" + "="*70)
        print("EXPECTED vs ACTUAL")
        print("="*70)
        
        expected = {
            'Detection F1': (0.98, results_df['f1'].mean()),
            'Fragmentation': (5.0, results_df['fragmentation'].mean()),
            'Track Length': (25, results_df['pred_mean_length'].mean())
        }
        
        for metric, (exp, act) in expected.items():
            status = "✅" if abs(act - exp) / exp < 0.2 else "⚠️"
            print(f"{status} {metric}: Expected ~{exp:.1f}, Got {act:.1f}")
    
    else:
        print("\n❌ No samples processed successfully")


if __name__ == '__main__':
    main()
