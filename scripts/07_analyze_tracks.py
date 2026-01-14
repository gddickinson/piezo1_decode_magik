#!/usr/bin/env python3
"""
Analyze Track Quality in Detail

Provides quantitative analysis of track lengths, gaps, and quality.

Usage:
    python scripts/07_analyze_tracks.py \
        --magik checkpoints/magik/best_model.pth \
        --decode checkpoints/decode_optimized/best_model.pth \
        --data data/synthetic_optimized \
        --output results/track_analysis \
        --num_samples 20 \
        --threshold 0.3
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.magik_gnn import MAGIKNet
from piezo1_magik.models.decode_net import DECODENet
from piezo1_magik.data.magik_dataset import MAGIKDataset


def greedy_linking(edge_probs, edge_index, threshold=0.5):
    """Greedy linking algorithm."""
    mask = edge_probs > threshold
    
    if mask.sum() == 0:
        return {}
    
    valid_edges = edge_index[:, mask]
    valid_probs = edge_probs[mask]
    
    sorted_idx = torch.argsort(valid_probs, descending=True)
    
    src_linked = set()
    dst_linked = set()
    links = []
    
    for idx in sorted_idx:
        src = valid_edges[0, idx].item()
        dst = valid_edges[1, idx].item()
        
        if src not in src_linked and dst not in dst_linked:
            links.append((src, dst))
            src_linked.add(src)
            dst_linked.add(dst)
    
    # Build tracks
    node_to_track = {}
    track_counter = 0
    
    forward_links = {}
    backward_links = {}
    
    for src, dst in links:
        forward_links[src] = dst
        backward_links[dst] = src
    
    all_nodes = set(forward_links.keys()) | set(backward_links.keys())
    track_starts = [n for n in all_nodes if n not in backward_links]
    
    for start_node in track_starts:
        if start_node in node_to_track:
            continue
        
        current = start_node
        while current is not None:
            node_to_track[current] = track_counter
            current = forward_links.get(current, None)
        
        track_counter += 1
    
    for node in all_nodes:
        if node not in node_to_track:
            node_to_track[node] = track_counter
            track_counter += 1
    
    return node_to_track


def analyze_track_lengths(pred_tracks, node_features):
    """
    Analyze predicted track lengths.
    
    Returns dict with track length statistics.
    """
    # Build track -> nodes mapping
    track_to_nodes = defaultdict(list)
    for node_idx, track_id in pred_tracks.items():
        track_to_nodes[track_id].append(node_idx)
    
    # Compute length of each track
    track_lengths = []
    track_temporal_spans = []
    track_gaps = []
    
    for track_id, node_indices in track_to_nodes.items():
        # Get frames for this track
        frames = [node_features[n, 0].item() for n in node_indices]
        frames_sorted = sorted(frames)
        
        # Track length = number of detections
        length = len(frames_sorted)
        track_lengths.append(length)
        
        # Temporal span = max frame - min frame + 1
        if len(frames_sorted) > 0:
            temporal_span = frames_sorted[-1] - frames_sorted[0] + 1
            track_temporal_spans.append(temporal_span)
            
            # Count gaps
            expected_frames = set(range(int(frames_sorted[0]), int(frames_sorted[-1]) + 1))
            actual_frames = set(int(f) for f in frames_sorted)
            num_gaps = len(expected_frames - actual_frames)
            track_gaps.append(num_gaps)
        else:
            track_temporal_spans.append(1)
            track_gaps.append(0)
    
    return {
        'lengths': track_lengths,
        'temporal_spans': track_temporal_spans,
        'gaps': track_gaps,
        'num_tracks': len(track_to_nodes)
    }


def analyze_gt_tracks(node_tracks, node_features):
    """Analyze ground truth track lengths."""
    gt_track_to_nodes = defaultdict(list)
    for node_idx in range(len(node_tracks)):
        track_id = node_tracks[node_idx].item()
        gt_track_to_nodes[track_id].append(node_idx)
    
    gt_lengths = []
    gt_temporal_spans = []
    
    for track_id, node_indices in gt_track_to_nodes.items():
        frames = [node_features[n, 0].item() for n in node_indices]
        frames_sorted = sorted(frames)
        
        gt_lengths.append(len(frames_sorted))
        if len(frames_sorted) > 0:
            temporal_span = frames_sorted[-1] - frames_sorted[0] + 1
            gt_temporal_spans.append(temporal_span)
    
    return {
        'lengths': gt_lengths,
        'temporal_spans': gt_temporal_spans,
        'num_tracks': len(gt_track_to_nodes)
    }


def analyze_fragmentation(pred_tracks, gt_tracks, node_features):
    """
    Analyze track fragmentation.
    
    For each GT track, count how many predicted tracks it's split into.
    """
    # Build mappings
    gt_track_to_nodes = defaultdict(list)
    for node_idx in range(len(gt_tracks)):
        track_id = gt_tracks[node_idx].item()
        gt_track_to_nodes[track_id].append(node_idx)
    
    # For each GT track, find which predicted tracks contain its nodes
    fragmentation_counts = []
    
    for gt_id, gt_nodes in gt_track_to_nodes.items():
        pred_track_ids = set()
        for node_idx in gt_nodes:
            if node_idx in pred_tracks:
                pred_track_ids.add(pred_tracks[node_idx])
        
        # Number of predicted tracks this GT track is split into
        fragmentation_counts.append(len(pred_track_ids))
    
    return fragmentation_counts


def analyze_sample(magik_model, sample, device, threshold):
    """Analyze one sample in detail."""
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
    
    # Get predicted tracks
    pred_tracks = greedy_linking(edge_probs.cpu(), edge_index.cpu(), threshold)
    
    # Analyze predicted tracks
    pred_analysis = analyze_track_lengths(pred_tracks, node_features.cpu())
    
    # Analyze GT tracks
    gt_analysis = analyze_gt_tracks(node_tracks.cpu(), node_features.cpu())
    
    # Analyze fragmentation
    fragmentation = analyze_fragmentation(pred_tracks, node_tracks.cpu(), node_features.cpu())
    
    return {
        'pred': pred_analysis,
        'gt': gt_analysis,
        'fragmentation': fragmentation
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze track quality')
    parser.add_argument('--magik', type=str, required=True)
    parser.add_argument('--decode', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/track_analysis')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=-1)
    
    args = parser.parse_args()
    
    print("="*70)
    print("TRACK QUALITY ANALYSIS")
    print("="*70)
    print(f"Threshold: {args.threshold}")
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
    print(f"✅ Analyzing {num_samples} samples\n")
    
    # Analyze all samples
    all_pred_lengths = []
    all_gt_lengths = []
    all_pred_spans = []
    all_gt_spans = []
    all_gaps = []
    all_fragmentation = []
    
    for i in tqdm(range(num_samples), desc='Analyzing'):
        sample = dataset[i]
        result = analyze_sample(magik_model, sample, device, args.threshold)
        
        if result is not None:
            all_pred_lengths.extend(result['pred']['lengths'])
            all_gt_lengths.extend(result['gt']['lengths'])
            all_pred_spans.extend(result['pred']['temporal_spans'])
            all_gt_spans.extend(result['gt']['temporal_spans'])
            all_gaps.extend(result['pred']['gaps'])
            all_fragmentation.extend(result['fragmentation'])
    
    # Print statistics
    print("\n" + "="*70)
    print("TRACK LENGTH ANALYSIS")
    print("="*70)
    
    print("\n📊 Predicted Track Lengths (# detections):")
    print(f"   Mean:   {np.mean(all_pred_lengths):.1f}")
    print(f"   Median: {np.median(all_pred_lengths):.1f}")
    print(f"   Min:    {np.min(all_pred_lengths)}")
    print(f"   Max:    {np.max(all_pred_lengths)}")
    print(f"   Std:    {np.std(all_pred_lengths):.1f}")
    
    print("\n📊 Ground Truth Track Lengths (# detections):")
    print(f"   Mean:   {np.mean(all_gt_lengths):.1f}")
    print(f"   Median: {np.median(all_gt_lengths):.1f}")
    print(f"   Min:    {np.min(all_gt_lengths)}")
    print(f"   Max:    {np.max(all_gt_lengths)}")
    
    print("\n⏱️  Predicted Track Temporal Spans (# frames):")
    print(f"   Mean:   {np.mean(all_pred_spans):.1f}")
    print(f"   Median: {np.median(all_pred_spans):.1f}")
    
    print("\n⏱️  Ground Truth Track Temporal Spans (# frames):")
    print(f"   Mean:   {np.mean(all_gt_spans):.1f}")
    print(f"   Median: {np.median(all_gt_spans):.1f}")
    
    print("\n🔍 Track Gaps (missing frames within tracks):")
    print(f"   Mean:   {np.mean(all_gaps):.1f}")
    print(f"   Median: {np.median(all_gaps):.1f}")
    print(f"   Max:    {np.max(all_gaps)}")
    
    print("\n💥 Fragmentation (# pred tracks per GT track):")
    print(f"   Mean:   {np.mean(all_fragmentation):.1f}")
    print(f"   Median: {np.median(all_fragmentation):.1f}")
    print(f"   Max:    {np.max(all_fragmentation)}")
    
    # Categorize tracks
    short_tracks = sum(1 for l in all_pred_lengths if l <= 3)
    medium_tracks = sum(1 for l in all_pred_lengths if 3 < l <= 10)
    long_tracks = sum(1 for l in all_pred_lengths if l > 10)
    
    print("\n📈 Track Length Distribution:")
    print(f"   Short (≤3):     {short_tracks} ({100*short_tracks/len(all_pred_lengths):.1f}%)")
    print(f"   Medium (4-10):  {medium_tracks} ({100*medium_tracks/len(all_pred_lengths):.1f}%)")
    print(f"   Long (>10):     {long_tracks} ({100*long_tracks/len(all_pred_lengths):.1f}%)")
    
    # Create plots
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("="*70)
    
    # Plot 1: Track length distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Track lengths
    ax = axes[0, 0]
    ax.hist(all_pred_lengths, bins=50, alpha=0.7, label='Predicted', color='#3498db', edgecolor='black')
    ax.hist(all_gt_lengths, bins=50, alpha=0.5, label='Ground Truth', color='orange', edgecolor='black')
    ax.axvline(np.mean(all_pred_lengths), color='blue', linestyle='--', linewidth=2,
              label=f'Pred Mean: {np.mean(all_pred_lengths):.1f}')
    ax.axvline(np.mean(all_gt_lengths), color='orange', linestyle='--', linewidth=2,
              label=f'GT Mean: {np.mean(all_gt_lengths):.1f}')
    ax.set_xlabel('Track Length (# detections)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Track Length Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temporal spans
    ax = axes[0, 1]
    ax.hist(all_pred_spans, bins=50, alpha=0.7, label='Predicted', color='#3498db', edgecolor='black')
    ax.hist(all_gt_spans, bins=50, alpha=0.5, label='Ground Truth', color='orange', edgecolor='black')
    ax.axvline(np.mean(all_pred_spans), color='blue', linestyle='--', linewidth=2,
              label=f'Pred Mean: {np.mean(all_pred_spans):.1f}')
    ax.axvline(np.mean(all_gt_spans), color='orange', linestyle='--', linewidth=2,
              label=f'GT Mean: {np.mean(all_gt_spans):.1f}')
    ax.set_xlabel('Temporal Span (# frames)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Temporal Span Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gaps
    ax = axes[1, 0]
    ax.hist(all_gaps, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax.axvline(np.mean(all_gaps), color='darkred', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(all_gaps):.1f}')
    ax.set_xlabel('Number of Gaps (missing frames)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Track Gaps Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Fragmentation
    ax = axes[1, 1]
    ax.hist(all_fragmentation, bins=range(1, max(all_fragmentation)+2), 
           alpha=0.7, color='#9b59b6', edgecolor='black')
    ax.axvline(np.mean(all_fragmentation), color='purple', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(all_fragmentation):.1f}')
    ax.set_xlabel('# Predicted Tracks per GT Track', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Track Fragmentation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'track_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_dir / 'track_length_analysis.png'}")
    
    # Plot 2: Diagnostic summary
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    summary_text = f"""
TRACK QUALITY DIAGNOSTIC SUMMARY
Threshold: {args.threshold}
{'='*50}

PREDICTED TRACKS:
  Mean Length:        {np.mean(all_pred_lengths):.1f} detections
  Median Length:      {np.median(all_pred_lengths):.1f} detections
  Mean Temporal Span: {np.mean(all_pred_spans):.1f} frames
  
GROUND TRUTH TRACKS:
  Mean Length:        {np.mean(all_gt_lengths):.1f} detections
  Mean Temporal Span: {np.mean(all_gt_spans):.1f} frames

TRACK QUALITY:
  Mean Gaps:          {np.mean(all_gaps):.1f} frames/track
  Mean Fragmentation: {np.mean(all_fragmentation):.1f} pred tracks/GT track
  
TRACK CATEGORIES:
  Short (≤3):         {short_tracks} ({100*short_tracks/len(all_pred_lengths):.1f}%)
  Medium (4-10):      {medium_tracks} ({100*medium_tracks/len(all_pred_lengths):.1f}%)
  Long (>10):         {long_tracks} ({100*long_tracks/len(all_pred_lengths):.1f}%)

DIAGNOSIS:
"""
    
    # Add diagnosis
    if np.mean(all_fragmentation) > 5:
        summary_text += "\n  ⚠️  HIGH FRAGMENTATION: GT tracks split into many pieces"
        summary_text += "\n      → Increase max_temporal_gap or max_spatial_distance"
    elif np.mean(all_pred_lengths) < 5:
        summary_text += "\n  ⚠️  VERY SHORT TRACKS: Most tracks are <5 detections"
        summary_text += "\n      → Linking algorithm creating too many fragments"
    elif short_tracks / len(all_pred_lengths) > 0.5:
        summary_text += "\n  ⚠️  MANY SHORT TRACKS: >50% are ≤3 detections"
        summary_text += "\n      → Graph connectivity issue or linking problem"
    else:
        summary_text += "\n  ✅ Track lengths look reasonable"
    
    if np.mean(all_gaps) > 10:
        summary_text += "\n  ⚠️  LARGE GAPS: Tracks have many missing frames"
        summary_text += "\n      → DECODE missing detections or particles blinking"
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.savefig(output_dir / 'diagnostic_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_dir / 'diagnostic_summary.png'}")
    
    # Save CSV
    results_df = pd.DataFrame({
        'pred_length': all_pred_lengths,
        'pred_temporal_span': all_pred_spans,
        'gaps': all_gaps
    })
    results_df.to_csv(output_dir / 'track_statistics.csv', index=False)
    
    fragmentation_df = pd.DataFrame({
        'fragmentation': all_fragmentation
    })
    fragmentation_df.to_csv(output_dir / 'fragmentation.csv', index=False)
    
    print(f"✅ Saved: {output_dir / 'track_statistics.csv'}")
    print(f"✅ Saved: {output_dir / 'fragmentation.csv'}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
