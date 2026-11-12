#!/usr/bin/env python3
"""
Complete DECODE + MAGIK Pipeline with Gap-Filling

Runs complete particle tracking pipeline on TIRF microscopy data.

Usage:
    python scripts/09_run_complete_pipeline.py \
        --input data/real_movie.tif \
        --decode checkpoints/decode_optimized/best_model.pth \
        --magik checkpoints/magik/best_model.pth \
        --output results/tracked \
        --threshold 0.3 \
        --use_gap_filling \
        --max_gap 40 \
        --max_distance 100
"""

import argparse
import torch
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.decode_net import DECODENet
from piezo1_magik.models.magik_gnn import MAGIKNet
from piezo1_magik.tracking.linking import (build_candidate_graph, build_track_fragments,
                                          fragments_to_node_mapping, greedy_linking, merge_track_fragments)


def load_models(decode_path, magik_path, device):
    """Load DECODE and MAGIK models."""
    print("Loading models...")

    # Load DECODE
    decode_checkpoint = torch.load(decode_path, map_location=device, weights_only=False)
    decode_config = decode_checkpoint.get('config', {})
    base_channels = decode_config.get('model', {}).get('base_channels', 32)

    decode_model = DECODENet(base_channels=base_channels)
    decode_model.load_state_dict(decode_checkpoint['model_state_dict'])
    decode_model = decode_model.to(device)
    decode_model.eval()

    # Load MAGIK
    magik_checkpoint = torch.load(magik_path, map_location=device, weights_only=False)
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

    print("✅ Models loaded")

    return decode_model, magik_model, magik_config


def run_decode(movie, decode_model, device, detection_threshold=0.5):
    """Run DECODE on all frames to detect particles."""
    print(f"\nRunning DECODE detection on {len(movie)} frames...")

    # Pad movie for temporal context (repeat first and last frames)
    padded_movie = np.concatenate([
        movie[0:1],  # Repeat first frame
        movie,
        movie[-1:]   # Repeat last frame
    ], axis=0)

    all_detections = []

    for frame_idx in tqdm(range(len(movie)), desc='Detecting'):
        # Get 3 consecutive frames (current frame is in the middle)
        # padded_movie[frame_idx] = movie[frame_idx - 1]
        # padded_movie[frame_idx + 1] = movie[frame_idx]  (the actual frame we want)
        # padded_movie[frame_idx + 2] = movie[frame_idx + 1]
        three_frames = padded_movie[frame_idx:frame_idx+3]  # Shape: (3, H, W)

        # Prepare tensor: (1, 3, H, W)
        frame_tensor = torch.from_numpy(three_frames).float().unsqueeze(0)
        frame_tensor = frame_tensor.to(device)

        # Run DECODE
        with torch.no_grad():
            outputs = decode_model(frame_tensor)

        # Extract detections
        prob_map = outputs['prob'][0, 0].cpu().numpy()
        coord_map = outputs['offset'][0].cpu().numpy()

        # Find peaks
        mask = prob_map > detection_threshold

        if mask.sum() > 0:
            # Get coordinates
            y_indices, x_indices = np.where(mask)

            for y, x in zip(y_indices, x_indices):
                # Get sub-pixel coordinates
                dx = coord_map[0, y, x]
                dy = coord_map[1, y, x]

                x_sub = x + dx
                y_sub = y + dy

                prob = prob_map[y, x]

                # Estimate photons (crude approximation)
                photons = prob * 1000

                all_detections.append({
                    'frame': frame_idx,
                    'x': x_sub,
                    'y': y_sub,
                    'probability': prob,
                    'photons': photons
                })

    detections_df = pd.DataFrame(all_detections)
    print(f"✅ Detected {len(detections_df)} particles across {len(movie)} frames")

    return detections_df


def build_graph(detections_df, max_temporal_gap, max_spatial_distance):
    """Build temporal graph from detections."""
    print(f"\nBuilding graph (temporal_gap≤{max_temporal_gap}, spatial_dist≤{max_spatial_distance})...")

    node_features, edge_index, edge_features = build_candidate_graph(
        detections_df['frame'].values, detections_df['x'].values, detections_df['y'].values,
        detections_df['probability'].values, detections_df['photons'].values,
        max_temporal_gap, max_spatial_distance)
    num_nodes = len(node_features)
    if edge_index.shape[1] == 0:
        print("⚠️  No edges created!")

    print(f"✅ Graph built: {num_nodes} nodes, {edge_index.shape[1]} edges")

    return node_features, edge_index, edge_features


def run_magik(node_features, edge_index, edge_features, magik_model, device, threshold):
    """Run MAGIK to predict links."""
    print(f"\nRunning MAGIK link prediction (threshold={threshold})...")

    if edge_index.shape[1] == 0:
        print("⚠️  No edges to process")
        return {}

    # Convert to tensors
    node_features_t = torch.from_numpy(node_features).float().to(device)
    edge_index_t = torch.from_numpy(edge_index).long().to(device)
    edge_features_t = torch.from_numpy(edge_features).float().to(device)

    # Run MAGIK
    with torch.no_grad():
        edge_logits = magik_model(node_features_t, edge_index_t, edge_features_t)
        edge_probs = torch.sigmoid(edge_logits)

    # Greedy one-to-one linking
    node_to_track = greedy_linking(edge_probs.cpu(), edge_index_t.cpu(), threshold)
    if not node_to_track:
        print("⚠️  No edges above threshold")
        return {}
    track_counter = len(set(node_to_track.values()))

    print(f"✅ Created {track_counter} initial track fragments")

    return node_to_track


def fill_gaps(node_to_track, node_features, max_gap, max_distance):
    """Fill gaps between track fragments."""
    print(f"\nFilling gaps (max_gap={max_gap}, max_distance={max_distance})...")

    fragments = build_track_fragments(node_to_track, node_features)
    print(f"  Initial fragments: {len(fragments)}")

    final_fragments, iteration = merge_track_fragments(fragments, max_gap, max_distance)

    print(f"  Final tracks: {len(final_fragments)}")
    print(f"  Merged {len(fragments) - len(final_fragments)} fragments in {iteration} iterations")

    node_to_track_merged = fragments_to_node_mapping(final_fragments)

    return node_to_track_merged


def save_tracks(detections_df, node_to_track, output_dir):
    """Save tracking results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add track IDs to detections
    track_ids = [-1] * len(detections_df)
    for node_idx, track_id in node_to_track.items():
        track_ids[node_idx] = track_id

    detections_df['track_id'] = track_ids

    # Save full tracks
    tracks_df = detections_df[detections_df['track_id'] >= 0].copy()
    tracks_df = tracks_df.sort_values(['track_id', 'frame'])
    tracks_df.to_csv(output_dir / 'tracks.csv', index=False)

    # Save summary
    track_stats = []
    for track_id in tracks_df['track_id'].unique():
        track_data = tracks_df[tracks_df['track_id'] == track_id]

        track_stats.append({
            'track_id': track_id,
            'length': len(track_data),
            'start_frame': track_data['frame'].min(),
            'end_frame': track_data['frame'].max(),
            'duration': track_data['frame'].max() - track_data['frame'].min() + 1,
            'mean_x': track_data['x'].mean(),
            'mean_y': track_data['y'].mean()
        })

    stats_df = pd.DataFrame(track_stats)
    stats_df.to_csv(output_dir / 'track_summary.csv', index=False)

    print(f"\n✅ Saved {len(tracks_df)} detections in {len(stats_df)} tracks")
    print(f"   Mean track length: {stats_df['length'].mean():.1f} detections")
    print(f"   Mean track duration: {stats_df['duration'].mean():.1f} frames")

    return tracks_df, stats_df


def visualize_tracks(movie, tracks_df, stats_df, output_dir):
    """Create visualization of tracking results."""
    output_dir = Path(output_dir)

    print("\nCreating visualizations...")

    # Pick middle frame
    middle_frame = len(movie) // 2

    # Create overlay
    fig, ax = plt.subplots(figsize=(12, 10))

    frame_img = movie[middle_frame]
    vmin, vmax = np.percentile(frame_img, [1, 99])
    ax.imshow(frame_img, cmap='gray', vmin=vmin, vmax=vmax)

    # Plot tracks visible in this frame range
    frame_range = range(max(0, middle_frame - 5), min(len(movie), middle_frame + 6))

    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for track_id in tracks_df['track_id'].unique():
        track_data = tracks_df[tracks_df['track_id'] == track_id]
        track_data_visible = track_data[track_data['frame'].isin(frame_range)]

        if len(track_data_visible) == 0:
            continue

        color = colors[track_id % 20]

        # Plot points
        ax.scatter(track_data_visible['x'], track_data_visible['y'],
                  s=80, facecolors='none', edgecolors=color, linewidths=2, alpha=0.8)

        # Plot trajectory
        track_sorted = track_data_visible.sort_values('frame')
        ax.plot(track_sorted['x'], track_sorted['y'], '-',
               color=color, linewidth=1.5, alpha=0.6)

    ax.set_title(f'Particle Tracks (Frame {middle_frame} ± 5)\n{len(stats_df)} total tracks',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'tracking_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Track statistics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(stats_df['length'], bins=30, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0].axvline(stats_df['length'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {stats_df['length'].mean():.1f}")
    axes[0].set_xlabel('Track Length (# detections)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Track Length Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(stats_df['duration'], bins=30, edgecolor='black', alpha=0.7, color='#2ecc71')
    axes[1].axvline(stats_df['duration'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {stats_df['duration'].mean():.1f}")
    axes[1].set_xlabel('Track Duration (# frames)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Track Duration Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'track_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualizations saved")


def main():
    parser = argparse.ArgumentParser(description='Complete DECODE+MAGIK pipeline')
    parser.add_argument('--input', type=str, required=True,
                       help='Input TIRF movie (.tif)')
    parser.add_argument('--decode', type=str, required=True,
                       help='Trained DECODE model')
    parser.add_argument('--magik', type=str, required=True,
                       help='Trained MAGIK model')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='MAGIK edge threshold')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                       help='DECODE detection threshold')
    parser.add_argument('--use_gap_filling', action='store_true',
                       help='Enable gap-filling post-processing')
    parser.add_argument('--max_gap', type=int, default=40,
                       help='Max frame gap for gap-filling')
    parser.add_argument('--max_distance', type=float, default=100,
                       help='Max spatial distance for gap-filling (pixels)')
    parser.add_argument('--gpu', type=int, default=-1)

    args = parser.parse_args()

    print("="*70)
    print("DECODE + MAGIK TRACKING PIPELINE")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"MAGIK threshold: {args.threshold}")
    print(f"Gap-filling: {'Enabled' if args.use_gap_filling else 'Disabled'}")
    if args.use_gap_filling:
        print(f"  Max gap: {args.max_gap} frames")
        print(f"  Max distance: {args.max_distance} pixels")
    print("="*70)

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")

    # Load movie
    print(f"Loading movie: {args.input}")
    movie = tifffile.imread(args.input)
    print(f"✅ Loaded movie: {movie.shape} (frames × height × width)\n")

    # Load models
    decode_model, magik_model, magik_config = load_models(
        args.decode, args.magik, device
    )

    # Run DECODE
    detections_df = run_decode(movie, decode_model, device, args.detection_threshold)

    # Build graph
    node_features, edge_index, edge_features = build_graph(
        detections_df,
        magik_config['data']['max_temporal_gap'],
        magik_config['data']['max_spatial_distance']
    )

    # Run MAGIK
    node_to_track = run_magik(
        node_features, edge_index, edge_features,
        magik_model, device, args.threshold
    )

    # Gap-filling
    if args.use_gap_filling and len(node_to_track) > 0:
        node_to_track = fill_gaps(
            node_to_track, node_features,
            args.max_gap, args.max_distance
        )

    # Save results
    tracks_df, stats_df = save_tracks(detections_df, node_to_track, args.output)

    # Visualize
    visualize_tracks(movie, tracks_df, stats_df, args.output)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Results saved to: {args.output}")
    print(f"  - tracks.csv (all detections with track IDs)")
    print(f"  - track_summary.csv (statistics per track)")
    print(f"  - tracking_overlay.png (visual overlay)")
    print(f"  - track_statistics.png (histograms)")


if __name__ == '__main__':
    main()
