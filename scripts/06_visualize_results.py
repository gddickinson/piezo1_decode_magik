#!/usr/bin/env python3
"""
Visualize PIEZO1 Analysis Results

Creates publication-quality visualizations of:
- Tracked puncta overlaid on movie
- Calcium fluorescence traces
- Event detection
- Track statistics

Usage:
    python 06_visualize_results.py \\
        --input results/ \\
        --movie dual_channel_movie.tif \\
        --output figures/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tifffile
from matplotlib import animation


def plot_tracks_overlay(movie, tracks_df, output_path, max_tracks=50):
    """Plot tracks overlaid on movie frame."""
    
    # Use first frame
    frame = movie[0] if movie.ndim == 4 else movie[0, 0]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(frame, cmap='gray', vmin=np.percentile(frame, 1), 
             vmax=np.percentile(frame, 99.9))
    
    # Plot tracks
    track_ids = tracks_df['track_id'].unique()[:max_tracks]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(track_ids)))
    
    for track_id, color in zip(track_ids, colors):
        track = tracks_df[tracks_df['track_id'] == track_id]
        ax.plot(track['x'], track['y'], '-', color=color, alpha=0.7, lw=2)
        ax.scatter(track['x'].iloc[0], track['y'].iloc[0], 
                  color=color, s=100, marker='o', edgecolor='white', lw=2)
    
    ax.set_title('PIEZO1 Puncta Tracks', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved tracks overlay: {output_path}")


def plot_calcium_traces(traces_df, events_df, output_path, max_tracks=10):
    """Plot calcium fluorescence traces."""
    
    track_ids = traces_df['track_id'].unique()[:max_tracks]
    
    fig, axes = plt.subplots(len(track_ids), 1, figsize=(12, 2*len(track_ids)), 
                            sharex=True)
    
    if len(track_ids) == 1:
        axes = [axes]
    
    for ax, track_id in zip(axes, track_ids):
        # Get trace
        trace = traces_df[traces_df['track_id'] == track_id]
        
        # Plot ΔF/F₀
        ax.plot(trace['frame'], trace['dff'], 'k-', lw=1)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        
        # Mark events
        track_events = events_df[events_df['track_id'] == track_id]
        for _, event in track_events.iterrows():
            ax.axvline(event['peak_frame'], color='r', alpha=0.5, ls='--')
            ax.text(event['peak_frame'], event['amplitude'], 
                   f"{event['amplitude']:.2f}", 
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('ΔF/F₀', fontsize=10)
        ax.set_title(f'Track {track_id}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Frame', fontsize=12)
    fig.suptitle('Calcium Fluorescence Traces', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved calcium traces: {output_path}")


def plot_event_statistics(events_df, output_path):
    """Plot event statistics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Amplitude distribution
    axes[0, 0].hist(events_df['amplitude'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Event Amplitude (ΔF/F₀)', fontsize=10)
    axes[0, 0].set_ylabel('Count', fontsize=10)
    axes[0, 0].set_title('Event Amplitude Distribution', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Duration distribution
    axes[0, 1].hist(events_df['duration'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Event Duration (frames)', fontsize=10)
    axes[0, 1].set_ylabel('Count', fontsize=10)
    axes[0, 1].set_title('Event Duration Distribution', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Events per track
    events_per_track = events_df.groupby('track_id').size()
    axes[1, 0].hist(events_per_track, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Events per Track', fontsize=10)
    axes[1, 0].set_ylabel('Count', fontsize=10)
    axes[1, 0].set_title('Event Frequency Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Amplitude vs Duration
    axes[1, 1].scatter(events_df['duration'], events_df['amplitude'], 
                      alpha=0.5, s=30)
    axes[1, 1].set_xlabel('Duration (frames)', fontsize=10)
    axes[1, 1].set_ylabel('Amplitude (ΔF/F₀)', fontsize=10)
    axes[1, 1].set_title('Event Amplitude vs Duration', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved event statistics: {output_path}")


def plot_track_statistics(tracks_df, output_path):
    """Plot track statistics."""
    
    # Compute statistics
    track_lengths = tracks_df.groupby('track_id').size()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Track length distribution
    axes[0].hist(track_lengths, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Track Length (frames)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Track Length Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Summary statistics
    stats_text = f"""
    Total Tracks: {len(track_lengths):,}
    Mean Length: {track_lengths.mean():.1f} frames
    Median Length: {track_lengths.median():.1f} frames
    Max Length: {track_lengths.max()} frames
    """
    
    axes[1].text(0.1, 0.5, stats_text, fontsize=12, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved track statistics: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize analysis results')
    parser.add_argument('--input', type=str, required=True,
                        help='Results directory')
    parser.add_argument('--movie', type=str, required=True,
                        help='Original movie file')
    parser.add_argument('--output', type=str, default='figures',
                        help='Output directory for figures')
    parser.add_argument('--max_tracks', type=int, default=50,
                        help='Maximum tracks to visualize')
    
    args = parser.parse_args()
    
    print("="*70)
    print("VISUALIZING RESULTS")
    print("="*70)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    tracks_df = pd.read_csv(input_dir / 'tracks.csv')
    print(f"✅ Loaded {len(tracks_df)} track points")
    
    # Load movie
    print(f"Loading movie: {args.movie}")
    movie = tifffile.imread(args.movie)
    print(f"✅ Movie shape: {movie.shape}")
    
    # Plot tracks
    print("\nCreating visualizations...")
    plot_tracks_overlay(
        movie, tracks_df, 
        output_dir / 'tracks_overlay.png',
        max_tracks=args.max_tracks
    )
    
    # Plot track statistics
    plot_track_statistics(tracks_df, output_dir / 'track_statistics.png')
    
    # Load calcium data if available
    traces_file = input_dir / 'calcium_traces.csv'
    events_file = input_dir / 'calcium_events.csv'
    
    if traces_file.exists() and events_file.exists():
        traces_df = pd.read_csv(traces_file)
        events_df = pd.read_csv(events_file)
        
        print(f"✅ Loaded {len(traces_df)} trace points")
        print(f"✅ Loaded {len(events_df)} events")
        
        # Plot calcium traces
        plot_calcium_traces(
            traces_df, events_df,
            output_dir / 'calcium_traces.png',
            max_tracks=min(10, args.max_tracks)
        )
        
        # Plot event statistics
        plot_event_statistics(events_df, output_dir / 'event_statistics.png')
    else:
        print("⚠️  No calcium data found, skipping calcium visualizations")
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
