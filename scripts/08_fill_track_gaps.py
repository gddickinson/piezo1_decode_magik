#!/usr/bin/env python3
"""
Gap-Filling Post-Processor for Track Fragments

Merges short track fragments into longer tracks by bridging temporal gaps.

Usage:
    python scripts/08_fill_track_gaps.py \
        --magik checkpoints/magik/best_model.pth \
        --decode checkpoints/decode_optimized/best_model.pth \
        --data data/synthetic_optimized \
        --output results/gap_filled_tracks \
        --num_samples 20 \
        --threshold 0.3 \
        --max_gap 10 \
        --max_distance 30
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.magik_gnn import MAGIKNet
from piezo1_magik.models.decode_net import DECODENet
from piezo1_magik.data.magik_dataset import MAGIKDataset
from piezo1_magik.tracking.linking import (build_track_fragments, fragments_to_node_mapping,
                                          greedy_linking, merge_track_fragments)
from piezo1_magik.tracking.track_stats import analyze_fragmentation, analyze_track_lengths


def process_sample(magik_model, sample, device, threshold, max_gap, max_distance):
    """Process one sample with gap filling."""
    node_features = sample['node_features'].to(device)
    edge_index = sample['edge_index'].to(device)
    edge_features = sample['edge_features'].to(device)
    node_tracks = sample['node_tracks'].to(device)
    
    if edge_index.shape[1] == 0:
        return None
    
    # Run MAGIK
    with torch.no_grad():
        edge_logits = magik_model(node_features, edge_index, edge_features)
        edge_probs = torch.sigmoid(edge_logits)
    
    # Get initial fragmented tracks
    initial_tracks = greedy_linking(edge_probs.cpu(), edge_index.cpu(), threshold)
    
    # Build fragments
    fragments = build_track_fragments(initial_tracks, node_features.cpu())
    
    # Merge fragments
    merged_fragments, _ = merge_track_fragments(fragments, max_gap, max_distance)
    
    # Convert back to node mapping
    merged_tracks = fragments_to_node_mapping(merged_fragments)
    
    # Analyze both
    initial_analysis = analyze_track_lengths(initial_tracks, node_features.cpu())
    merged_analysis = analyze_track_lengths(merged_tracks, node_features.cpu())
    
    initial_frag = analyze_fragmentation(initial_tracks, node_tracks.cpu(), node_features.cpu())
    merged_frag = analyze_fragmentation(merged_tracks, node_tracks.cpu(), node_features.cpu())
    
    return {
        'initial': initial_analysis,
        'merged': merged_analysis,
        'initial_frag': initial_frag,
        'merged_frag': merged_frag
    }


def main():
    parser = argparse.ArgumentParser(description='Gap-filling post-processor')
    parser.add_argument('--magik', type=str, required=True)
    parser.add_argument('--decode', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/gap_filled')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--max_gap', type=int, default=10,
                       help='Maximum frame gap to bridge')
    parser.add_argument('--max_distance', type=float, default=30,
                       help='Maximum spatial distance for merging (pixels)')
    parser.add_argument('--gpu', type=int, default=-1)
    
    args = parser.parse_args()
    
    print("="*70)
    print("GAP-FILLING POST-PROCESSOR")
    print("="*70)
    print(f"Threshold: {args.threshold}")
    print(f"Max gap: {args.max_gap} frames")
    print(f"Max distance: {args.max_distance} pixels")
    print("="*70)
    
    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")
    
    # Load models
    print("Loading models...")
    decode_checkpoint = torch.load(args.decode, map_location=device, weights_only=False)
    decode_config = decode_checkpoint.get('config', {})
    base_channels = decode_config.get('model', {}).get('base_channels', 32)
    
    decode_model = DECODENet(base_channels=base_channels)
    decode_model.load_state_dict(decode_checkpoint['model_state_dict'])
    decode_model = decode_model.to(device)
    decode_model.eval()
    
    magik_checkpoint = torch.load(args.magik, map_location=device, weights_only=False)
    magik_config = magik_checkpoint.get('config', {})
    
    magik_model = MAGIKNet(
        node_features=magik_config['model']['node_features'],
        edge_features=magik_config['model']['edge_features'],
        hidden_dim=magik_config['model']['hidden_dim'],
        num_layers=magik_config['model']['num_layers']
    )
    magik_model.load_state_dict(magik_checkpoint['model_state_dict'])
    magik_model = magik_model.to(device)
    magik_model.eval()
    
    print("✅ Models loaded\n")
    
    # Load dataset
    print("Creating dataset...")
    dataset = MAGIKDataset(
        args.data, decode_model, device,
        max_temporal_gap=magik_config['data']['max_temporal_gap'],
        max_spatial_distance=magik_config['data']['max_spatial_distance']
    )
    
    num_samples = min(args.num_samples, len(dataset))
    print(f"✅ Processing {num_samples} samples\n")
    
    # Process all samples
    all_initial_lengths = []
    all_merged_lengths = []
    all_initial_frag = []
    all_merged_frag = []
    
    for i in tqdm(range(num_samples), desc='Processing'):
        sample = dataset[i]
        result = process_sample(
            magik_model, sample, device, 
            args.threshold, args.max_gap, args.max_distance
        )
        
        if result is not None:
            all_initial_lengths.extend(result['initial']['lengths'])
            all_merged_lengths.extend(result['merged']['lengths'])
            all_initial_frag.extend(result['initial_frag'])
            all_merged_frag.extend(result['merged_frag'])
    
    # Print results
    print("\n" + "="*70)
    print("GAP-FILLING RESULTS")
    print("="*70)
    
    print("\n📊 BEFORE Gap Filling:")
    print(f"   Mean track length:  {np.mean(all_initial_lengths):.1f}")
    print(f"   Median track length: {np.median(all_initial_lengths):.1f}")
    print(f"   Number of tracks:    {len(all_initial_lengths)}")
    print(f"   Mean fragmentation:  {np.mean(all_initial_frag):.1f}×")
    
    print("\n📊 AFTER Gap Filling:")
    print(f"   Mean track length:  {np.mean(all_merged_lengths):.1f}")
    print(f"   Median track length: {np.median(all_merged_lengths):.1f}")
    print(f"   Number of tracks:    {len(all_merged_lengths)}")
    print(f"   Mean fragmentation:  {np.mean(all_merged_frag):.1f}×")
    
    print("\n🎯 Improvement:")
    improvement_length = np.mean(all_merged_lengths) / np.mean(all_initial_lengths)
    improvement_frag = np.mean(all_initial_frag) / np.mean(all_merged_frag)
    reduction_tracks = len(all_initial_lengths) / len(all_merged_lengths)
    
    print(f"   Track length:    {improvement_length:.2f}× longer")
    print(f"   Fragmentation:   {improvement_frag:.2f}× less fragmented")
    print(f"   Track count:     {reduction_tracks:.2f}× fewer tracks")
    
    # Create comparison plot
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Track lengths
    ax = axes[0, 0]
    ax.hist(all_initial_lengths, bins=50, alpha=0.7, label='Before', 
           color='#e74c3c', edgecolor='black')
    ax.hist(all_merged_lengths, bins=50, alpha=0.7, label='After',
           color='#2ecc71', edgecolor='black')
    ax.axvline(np.mean(all_initial_lengths), color='red', linestyle='--',
              label=f'Before Mean: {np.mean(all_initial_lengths):.1f}')
    ax.axvline(np.mean(all_merged_lengths), color='green', linestyle='--',
              label=f'After Mean: {np.mean(all_merged_lengths):.1f}')
    ax.set_xlabel('Track Length (# detections)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Track Length Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fragmentation
    ax = axes[0, 1]
    ax.hist(all_initial_frag, bins=range(1, max(all_initial_frag)+2),
           alpha=0.7, label='Before', color='#e74c3c', edgecolor='black')
    ax.hist(all_merged_frag, bins=range(1, max(max(all_merged_frag)+2, 10)),
           alpha=0.7, label='After', color='#2ecc71', edgecolor='black')
    ax.axvline(np.mean(all_initial_frag), color='red', linestyle='--',
              label=f'Before Mean: {np.mean(all_initial_frag):.1f}')
    ax.axvline(np.mean(all_merged_frag), color='green', linestyle='--',
              label=f'After Mean: {np.mean(all_merged_frag):.1f}')
    ax.set_xlabel('Fragmentation (# pred tracks per GT)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Fragmentation Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Track count categories
    ax = axes[1, 0]
    short_before = sum(1 for l in all_initial_lengths if l <= 3)
    medium_before = sum(1 for l in all_initial_lengths if 3 < l <= 10)
    long_before = sum(1 for l in all_initial_lengths if l > 10)
    
    short_after = sum(1 for l in all_merged_lengths if l <= 3)
    medium_after = sum(1 for l in all_merged_lengths if 3 < l <= 10)
    long_after = sum(1 for l in all_merged_lengths if l > 10)
    
    categories = ['Short\n(≤3)', 'Medium\n(4-10)', 'Long\n(>10)']
    before_counts = [short_before, medium_before, long_before]
    after_counts = [short_after, medium_after, long_after]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, before_counts, width, label='Before', 
          color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, after_counts, width, label='After',
          color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Track Categories', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
GAP-FILLING SUMMARY
Max Gap: {args.max_gap} frames
Max Distance: {args.max_distance} pixels
{'='*40}

BEFORE:
  Mean Length:       {np.mean(all_initial_lengths):.1f}
  Mean Fragmentation: {np.mean(all_initial_frag):.1f}×
  Track Count:       {len(all_initial_lengths)}

AFTER:
  Mean Length:       {np.mean(all_merged_lengths):.1f}
  Mean Fragmentation: {np.mean(all_merged_frag):.1f}×
  Track Count:       {len(all_merged_lengths)}

IMPROVEMENT:
  Length:      {improvement_length:.2f}× longer
  Fragmentation: {improvement_frag:.2f}× better
  Tracks:      {reduction_tracks:.2f}× fewer
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gap_filling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Plot saved: {output_dir / 'gap_filling_comparison.png'}")
    
    print("\n" + "="*70)
    print("GAP-FILLING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
