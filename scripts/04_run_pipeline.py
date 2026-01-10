#!/usr/bin/env python3
"""
Complete PIEZO1 Analysis Pipeline

Combines DECODE localization, MAGIK tracking, and calcium ROI analysis
into a single automated pipeline.

Pipeline:
1. Load dual-channel movie (PIEZO1 + Calcium)
2. Detect puncta with DECODE (frame-by-frame)
3. Link detections into tracks with MAGIK GNN
4. Extract calcium fluorescence traces at each track
5. Detect calcium events
6. Save all results and visualizations

Usage:
    python 04_run_pipeline.py \\
        --decode checkpoints/decode/best_model.pth \\
        --magik checkpoints/magik/best_model.pth \\
        --input /path/to/dual/channel/movie.tif \\
        --output results/
"""

import argparse
import torch
import numpy as np
import tifffile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.decode_net import DECODENet
from piezo1_magik.models.magik_gnn import MAGIKNet, build_tracking_graph
from piezo1_magik.analysis.calcium_roi import CalciumROIAnalyzer, analyze_all_tracks


def load_dual_channel_movie(movie_path):
    """
    Load dual-channel movie.
    
    Expected format: (2, T, H, W) or (T, 2, H, W)
    Channel 0: PIEZO1-HaloTag
    Channel 1: Calcium indicator
    
    Returns:
        piezo1_movie: (T, H, W)
        calcium_movie: (T, H, W)
    """
    movie = tifffile.imread(movie_path)
    
    if movie.ndim == 4:
        if movie.shape[0] == 2:
            # (2, T, H, W)
            piezo1_movie = movie[0]
            calcium_movie = movie[1]
        elif movie.shape[1] == 2:
            # (T, 2, H, W)
            piezo1_movie = movie[:, 0]
            calcium_movie = movie[:, 1]
        else:
            raise ValueError(f"Unexpected movie shape: {movie.shape}")
    elif movie.ndim == 3:
        # Single channel - assume PIEZO1 only
        print("⚠️  Warning: Single channel movie, no calcium analysis will be performed")
        piezo1_movie = movie
        calcium_movie = None
    else:
        raise ValueError(f"Unexpected movie dimensions: {movie.ndim}")
    
    return piezo1_movie, calcium_movie


def detect_puncta_decode(piezo1_movie, decode_model, device, detection_threshold=0.5):
    """
    Detect puncta using DECODE model.
    
    Args:
        piezo1_movie: (T, H, W) PIEZO1 channel
        decode_model: Trained DECODE model
        device: torch device
        detection_threshold: Detection threshold
        
    Returns:
        detections_per_frame: List[List[Dict]] detections per frame
    """
    T, H, W = piezo1_movie.shape
    
    # Normalize
    piezo1_norm = piezo1_movie.astype(np.float32) / 65535.0
    
    detections_per_frame = []
    
    decode_model.eval()
    
    print("Detecting puncta with DECODE...")
    with torch.no_grad():
        for t in tqdm(range(1, T-1), desc='DECODE'):
            # Extract 3-frame window
            window = piezo1_norm[t-1:t+2]  # (3, H, W)
            
            # Add batch dimension
            window_tensor = torch.from_numpy(window).unsqueeze(0).to(device)  # (1, 3, H, W)
            
            # Detect
            batch_detections = decode_model.predict(window_tensor, threshold=detection_threshold)
            
            detections_per_frame.append(batch_detections[0])
    
    # Add empty lists for first and last frames (no 3-frame window)
    detections_per_frame = [[]] + detections_per_frame + [[]]
    
    total_detections = sum(len(dets) for dets in detections_per_frame)
    print(f"✅ Detected {total_detections} puncta across {T} frames")
    
    return detections_per_frame


def track_with_magik(detections_per_frame, magik_model, device,
                     max_frame_gap=5, max_distance=10.0, link_threshold=0.5):
    """
    Link detections into tracks using MAGIK GNN.
    
    Args:
        detections_per_frame: Detections from DECODE
        magik_model: Trained MAGIK model
        device: torch device
        max_frame_gap: Max frames to link across
        max_distance: Max spatial distance (pixels)
        link_threshold: Probability threshold for linking
        
    Returns:
        tracks_df: DataFrame with tracks
    """
    print("\nBuilding tracking graph...")
    
    # Build graph
    node_features, edge_index, edge_features, node_map = build_tracking_graph(
        detections_per_frame,
        max_frame_gap=max_frame_gap,
        max_distance=max_distance
    )
    
    print(f"  Nodes: {node_features.shape[0]}")
    print(f"  Candidate edges: {edge_index.shape[1]}")
    
    if edge_index.shape[1] == 0:
        print("⚠️  No candidate edges - returning empty tracks")
        return pd.DataFrame(columns=['track_id', 'frame', 'x', 'y', 'photons'])
    
    # Move to device
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    edge_features = edge_features.to(device)
    
    # Predict links
    print("Predicting links with MAGIK...")
    magik_model.eval()
    with torch.no_grad():
        links, link_probs = magik_model.predict_links(
            node_features, edge_index, edge_features, threshold=link_threshold
        )
    
    print(f"✅ Predicted {len(links)} links")
    
    # Convert links to tracks using connected components
    tracks = assign_track_ids(links, len(node_features))
    
    # Create DataFrame
    track_data = []
    for node_idx, track_id in enumerate(tracks):
        if track_id >= 0:  # Valid track
            frame_idx, det_idx = node_map[node_idx]
            detection = detections_per_frame[frame_idx][det_idx]
            
            track_data.append({
                'track_id': track_id,
                'frame': frame_idx,
                'x': detection['x'],
                'y': detection['y'],
                'photons': detection['photons'],
                'sigma_x': detection['sigma_x'],
                'sigma_y': detection['sigma_y']
            })
    
    tracks_df = pd.DataFrame(track_data)
    
    num_tracks = tracks_df['track_id'].nunique() if len(tracks_df) > 0 else 0
    print(f"✅ Created {num_tracks} tracks")
    
    return tracks_df


def assign_track_ids(links, num_nodes):
    """
    Assign track IDs using connected components.
    
    Args:
        links: (N, 2) array of linked node pairs
        num_nodes: Total number of nodes
        
    Returns:
        track_ids: (num_nodes,) array of track IDs (-1 for unlinked)
    """
    from collections import defaultdict
    
    # Build adjacency list
    graph = defaultdict(set)
    for i, j in links:
        graph[i].add(j)
        graph[j].add(i)
    
    # Find connected components
    track_ids = np.full(num_nodes, -1)
    current_track_id = 0
    visited = set()
    
    for node in range(num_nodes):
        if node in visited:
            continue
        
        # BFS to find connected component
        component = []
        queue = [node]
        visited.add(node)
        
        while queue:
            current = queue.pop(0)
            component.append(current)
            
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Assign track ID to entire component
        for n in component:
            track_ids[n] = current_track_id
        
        current_track_id += 1
    
    return track_ids


def analyze_calcium(calcium_movie, tracks_df, roi_size=5):
    """
    Analyze calcium signals at tracked puncta.
    
    Args:
        calcium_movie: (T, H, W) calcium channel
        tracks_df: Tracks DataFrame
        roi_size: ROI size for extraction
        
    Returns:
        traces_df: Fluorescence traces
        events_df: Detected events
    """
    print("\nAnalyzing calcium signals...")
    
    traces_df, events_df = analyze_all_tracks(
        calcium_movie,
        tracks_df,
        roi_size=roi_size
    )
    
    print(f"✅ Extracted {len(traces_df)} trace points")
    print(f"✅ Detected {len(events_df)} calcium events")
    
    return traces_df, events_df


def save_results(output_dir, tracks_df, traces_df, events_df, detections_per_frame):
    """Save all results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tracks
    tracks_csv = output_dir / 'tracks.csv'
    tracks_df.to_csv(tracks_csv, index=False)
    print(f"\n💾 Saved tracks: {tracks_csv}")
    
    # Save calcium traces
    if len(traces_df) > 0:
        traces_csv = output_dir / 'calcium_traces.csv'
        traces_df.to_csv(traces_csv, index=False)
        print(f"💾 Saved calcium traces: {traces_csv}")
    
    # Save calcium events
    if len(events_df) > 0:
        events_csv = output_dir / 'calcium_events.csv'
        events_df.to_csv(events_csv, index=False)
        print(f"💾 Saved calcium events: {events_csv}")
    
    # Save summary statistics
    summary = {
        'num_frames': len(detections_per_frame),
        'total_detections': sum(len(d) for d in detections_per_frame),
        'num_tracks': int(tracks_df['track_id'].nunique()) if len(tracks_df) > 0 else 0,
        'num_calcium_events': len(events_df),
        'avg_track_length': float(tracks_df.groupby('track_id').size().mean()) if len(tracks_df) > 0 else 0
    }
    
    # Per-track statistics
    if len(tracks_df) > 0:
        track_stats = []
        for track_id in tracks_df['track_id'].unique():
            track = tracks_df[tracks_df['track_id'] == track_id]
            
            track_events = events_df[events_df['track_id'] == track_id] if len(events_df) > 0 else []
            
            track_stats.append({
                'track_id': int(track_id),
                'length': len(track),
                'start_frame': int(track['frame'].min()),
                'end_frame': int(track['frame'].max()),
                'num_events': len(track_events)
            })
        
        track_stats_df = pd.DataFrame(track_stats)
        track_stats_csv = output_dir / 'track_statistics.csv'
        track_stats_df.to_csv(track_stats_csv, index=False)
        print(f"💾 Saved track statistics: {track_stats_csv}")
    
    summary_json = output_dir / 'summary.json'
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved summary: {summary_json}")


def main():
    parser = argparse.ArgumentParser(description='Run complete PIEZO1 analysis pipeline')
    parser.add_argument('--decode', type=str, required=True,
                        help='Path to DECODE model checkpoint')
    parser.add_argument('--magik', type=str, required=True,
                        help='Path to MAGIK model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input dual-channel movie (TIFF)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                        help='DECODE detection threshold')
    parser.add_argument('--link_threshold', type=float, default=0.5,
                        help='MAGIK linking threshold')
    parser.add_argument('--max_distance', type=float, default=10.0,
                        help='Maximum linking distance (pixels)')
    parser.add_argument('--roi_size', type=int, default=5,
                        help='ROI size for calcium extraction')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PIEZO1 DECODE-MAGIK PIPELINE")
    print("="*70)
    
    # Setup device
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"\nDevice: {device}")
    
    # Load models
    print(f"\nLoading DECODE model: {args.decode}")
    decode_checkpoint = torch.load(args.decode, map_location=device, weights_only=False)
    decode_model = DECODENet(base_channels=32)
    decode_model.load_state_dict(decode_checkpoint['model_state_dict'])
    decode_model = decode_model.to(device)
    decode_model.eval()
    print("✅ DECODE loaded")
    
    print(f"\nLoading MAGIK model: {args.magik}")
    magik_checkpoint = torch.load(args.magik, map_location=device, weights_only=False)
    magik_model = MAGIKNet()
    magik_model.load_state_dict(magik_checkpoint['model_state_dict'])
    magik_model = magik_model.to(device)
    magik_model.eval()
    print("✅ MAGIK loaded")
    
    # Load movie
    print(f"\nLoading movie: {args.input}")
    piezo1_movie, calcium_movie = load_dual_channel_movie(args.input)
    print(f"✅ Movie loaded:")
    print(f"   PIEZO1 shape: {piezo1_movie.shape}")
    if calcium_movie is not None:
        print(f"   Calcium shape: {calcium_movie.shape}")
    
    # Stage 1: Detect puncta
    print(f"\n{'='*70}")
    print("STAGE 1: DECODE LOCALIZATION")
    print(f"{'='*70}")
    
    detections_per_frame = detect_puncta_decode(
        piezo1_movie,
        decode_model,
        device,
        detection_threshold=args.detection_threshold
    )
    
    # Stage 2: Track puncta
    print(f"\n{'='*70}")
    print("STAGE 2: MAGIK TRACKING")
    print(f"{'='*70}")
    
    tracks_df = track_with_magik(
        detections_per_frame,
        magik_model,
        device,
        max_distance=args.max_distance,
        link_threshold=args.link_threshold
    )
    
    # Stage 3: Calcium analysis
    if calcium_movie is not None and len(tracks_df) > 0:
        print(f"\n{'='*70}")
        print("STAGE 3: CALCIUM ROI ANALYSIS")
        print(f"{'='*70}")
        
        traces_df, events_df = analyze_calcium(
            calcium_movie,
            tracks_df,
            roi_size=args.roi_size
        )
    else:
        traces_df = pd.DataFrame()
        events_df = pd.DataFrame()
        if calcium_movie is None:
            print("\n⚠️  Skipping calcium analysis (no calcium channel)")
        else:
            print("\n⚠️  Skipping calcium analysis (no tracks)")
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    save_results(args.output, tracks_df, traces_df, events_df, detections_per_frame)
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {args.output}")
    print(f"\nFiles created:")
    print(f"  - tracks.csv (track coordinates)")
    print(f"  - track_statistics.csv (per-track stats)")
    if len(traces_df) > 0:
        print(f"  - calcium_traces.csv (fluorescence traces)")
    if len(events_df) > 0:
        print(f"  - calcium_events.csv (detected events)")
    print(f"  - summary.json (overall statistics)")


if __name__ == '__main__':
    main()
