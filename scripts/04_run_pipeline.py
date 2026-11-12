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

    Every other setting comes from configs/pipeline.yaml (--config); options given here override it.
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
from piezo1_magik.models.magik_gnn import MAGIKNet
from piezo1_magik.tracking.linking import build_candidate_graph, greedy_linking
from piezo1_magik.pipeline_config import DEFAULT_CONFIG, resolve
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
                     max_frame_gap=2, max_distance=10.0, link_threshold=0.5):
    """
    Link detections into tracks using MAGIK GNN.

    Same method as 03c/07/08/09 (piezo1_magik.tracking.linking): candidate edges join detections
    1..max_frame_gap frames apart and within max_distance pixels; MAGIK gives each edge a probability
    (sigmoid of its logit); greedy_linking accepts edges above link_threshold, highest probability first,
    with at most one link into and one out of each detection. Detections with no accepted link are not
    part of any track.

    Args:
        detections_per_frame: Detections from DECODE
        magik_model: Trained MAGIK model
        device: torch device
        max_frame_gap: Max frames to link across; keep equal to the MAGIK training gap (max_temporal_gap,
            2 in configs/magik_training.yaml), since the gap is an edge feature the network was trained on
        max_distance: Max spatial distance (pixels)
        link_threshold: Probability threshold for linking

    Returns:
        tracks_df: DataFrame with tracks
    """
    print("\nBuilding tracking graph...")

    # Flatten the per-frame lists into graph nodes (frame order); node_map[i] = (frame, index in that frame)
    node_map = [(f, k) for f, dets in enumerate(detections_per_frame) for k in range(len(dets))]
    dets = [detections_per_frame[f][k] for f, k in node_map]
    node_features, edge_index, edge_features = build_candidate_graph(
        [f for f, _ in node_map], [float(d['x']) for d in dets], [float(d['y']) for d in dets],
        [float(d.get('prob', 1.0)) for d in dets], [float(d.get('photons', 1000.0)) for d in dets],
        max_frame_gap, max_distance)

    print(f"  Nodes: {node_features.shape[0]}")
    print(f"  Candidate edges: {edge_index.shape[1]}")

    empty = pd.DataFrame(columns=['track_id', 'frame', 'x', 'y', 'photons', 'sigma_x', 'sigma_y'])
    if edge_index.shape[1] == 0:
        print("⚠️  No candidate edges - returning empty tracks")
        return empty

    print("Predicting links with MAGIK...")
    magik_model.eval()
    with torch.no_grad():
        edge_logits = magik_model(torch.from_numpy(node_features).float().to(device),
                                  torch.from_numpy(edge_index).long().to(device),
                                  torch.from_numpy(edge_features).float().to(device))
        edge_probs = torch.sigmoid(edge_logits).cpu()

    node_to_track = greedy_linking(edge_probs, edge_index, link_threshold)
    print(f"✅ Linked {len(node_to_track)} of {len(node_map)} detections")
    if not node_to_track:
        return empty

    track_data = []
    for node_idx, track_id in sorted(node_to_track.items(), key=lambda kv: (kv[1], node_map[kv[0]][0])):
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
    print(f"✅ Created {tracks_df['track_id'].nunique()} tracks")
    return tracks_df


def analyze_calcium(calcium_movie, tracks_df, **calcium_options):
    """
    Analyze calcium signals at tracked puncta.

    Args:
        calcium_movie: (T, H, W) calcium channel
        tracks_df: Tracks DataFrame
        calcium_options: roi_size, baseline_method, baseline_percentile, event_threshold, min_duration,
            min_distance (passed to analyze_all_tracks)

    Returns:
        traces_df: Fluorescence traces
        events_df: Detected events
    """
    print("\nAnalyzing calcium signals...")
    traces_df, events_df = analyze_all_tracks(calcium_movie, tracks_df, **calcium_options)
    print(f"✅ Extracted {len(traces_df)} trace points")
    print(f"✅ Detected {len(events_df)} calcium events")
    return traces_df, events_df


def filter_min_photons(detections_per_frame, min_photons):
    """Drop detections whose DECODE photon estimate is below min_photons (detection.min_photons)."""
    return [[d for d in dets if float(d['photons']) >= min_photons] for dets in detections_per_frame]


def save_results(output_dir, tracks_df, traces_df, events_df, detections_per_frame,
                 save_tracks=True, save_calcium_traces=True, save_events=True):
    """Save results (output.save_* in pipeline.yaml choose the CSVs); summary.json is always written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tracks
    if save_tracks:
        tracks_csv = output_dir / 'tracks.csv'
        tracks_df.to_csv(tracks_csv, index=False)
        print(f"\n💾 Saved tracks: {tracks_csv}")
    
    # Save calcium traces
    if save_calcium_traces and len(traces_df) > 0:
        traces_csv = output_dir / 'calcium_traces.csv'
        traces_df.to_csv(traces_csv, index=False)
        print(f"💾 Saved calcium traces: {traces_csv}")
    
    # Save calcium events
    if save_events and len(events_df) > 0:
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


def build_parser():
    """Command-line options; each one left out takes its value from --config (configs/pipeline.yaml)."""
    parser = argparse.ArgumentParser(description='Run complete PIEZO1 analysis pipeline')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG),
                        help='Pipeline YAML (default configs/pipeline.yaml); options below override it')
    parser.add_argument('--input', type=str, required=True,
                        help='Input dual-channel movie (TIFF)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--decode', type=str, help='DECODE checkpoint (models.decode)')
    parser.add_argument('--magik', type=str, help='MAGIK checkpoint (models.magik)')
    parser.add_argument('--detection_threshold', type=float, help='DECODE detection threshold (detection.threshold)')
    parser.add_argument('--min_photons', type=float, help='Minimum photons per detection (detection.min_photons)')
    parser.add_argument('--max_frame_gap', type=int,
                        help='Maximum frames to link across (tracking.max_frame_gap; keep equal to the MAGIK '
                             'training gap, 2)')
    parser.add_argument('--max_distance', type=float,
                        help='Maximum linking distance (pixels) (tracking.max_distance)')
    parser.add_argument('--link_threshold', type=float, help='MAGIK linking threshold (tracking.link_threshold)')
    parser.add_argument('--roi_size', type=int, help='ROI size for calcium extraction (calcium.roi_size)')
    parser.add_argument('--baseline_method', type=str, choices=['percentile', 'rolling', 'mean'],
                        help='F0 method (calcium.baseline_method)')
    parser.add_argument('--baseline_percentile', type=float, help='F0 percentile (calcium.baseline_percentile)')
    parser.add_argument('--event_threshold', type=float,
                        help='Event threshold, x robust sigma above median dF/F0 (calcium.event_threshold)')
    parser.add_argument('--min_duration', type=int, help='Minimum event duration (frames) (calcium.min_duration)')
    parser.add_argument('--min_distance', type=int, help='Minimum frames between events (calcium.min_distance)')
    for name in ('save_tracks', 'save_calcium_traces', 'save_events'):
        parser.add_argument(f'--{name}', action=argparse.BooleanOptionalAction, default=None,
                            help=f'Write the CSV (output.{name})')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'], help='Preferred device (processing.device)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser


def pick_device(preferred, gpu=0):
    """The preferred device if available, else the best available one."""
    if preferred == 'cuda' and torch.cuda.is_available():
        return f'cuda:{gpu}'
    if preferred in ('cuda', 'mps') and torch.backends.mps.is_available():
        return 'mps'
    if preferred == 'cpu':
        return 'cpu'
    return f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'


def main(argv=None):
    args = build_parser().parse_args(argv)
    cli = {k: v for k, v in vars(args).items() if k not in ('config', 'input', 'output', 'gpu')}
    cfg = resolve(cli, args.config)
    print(f"Config: {args.config} (command-line options override it)")

    print("="*70)
    print("PIEZO1 DECODE-MAGIK PIPELINE")
    print("="*70)
    
    # Setup device
    device = pick_device(cfg['device'], args.gpu)
    print(f"\nDevice: {device}")
    
    # Load models
    print(f"\nLoading DECODE model: {cfg['decode']}")
    decode_checkpoint = torch.load(cfg['decode'], map_location=device, weights_only=False)
    decode_model = DECODENet(base_channels=32)
    decode_model.load_state_dict(decode_checkpoint['model_state_dict'])
    decode_model = decode_model.to(device)
    decode_model.eval()
    print("✅ DECODE loaded")
    
    print(f"\nLoading MAGIK model: {cfg['magik']}")
    magik_checkpoint = torch.load(cfg['magik'], map_location=device, weights_only=False)
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
        detection_threshold=cfg['detection_threshold']
    )
    detections_per_frame = filter_min_photons(detections_per_frame, cfg['min_photons'])
    
    # Stage 2: Track puncta
    print(f"\n{'='*70}")
    print("STAGE 2: MAGIK TRACKING")
    print(f"{'='*70}")
    
    tracks_df = track_with_magik(
        detections_per_frame,
        magik_model,
        device,
        max_frame_gap=cfg['max_frame_gap'],
        max_distance=cfg['max_distance'],
        link_threshold=cfg['link_threshold']
    )
    
    # Stage 3: Calcium analysis
    if calcium_movie is not None and len(tracks_df) > 0:
        print(f"\n{'='*70}")
        print("STAGE 3: CALCIUM ROI ANALYSIS")
        print(f"{'='*70}")
        
        traces_df, events_df = analyze_calcium(
            calcium_movie,
            tracks_df,
            **{k: cfg[k] for k in ('roi_size', 'baseline_method', 'baseline_percentile', 'event_threshold',
                                   'min_duration', 'min_distance')}
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
    
    save_results(args.output, tracks_df, traces_df, events_df, detections_per_frame,
                 save_tracks=cfg['save_tracks'], save_calcium_traces=cfg['save_calcium_traces'],
                 save_events=cfg['save_events'])
    
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
