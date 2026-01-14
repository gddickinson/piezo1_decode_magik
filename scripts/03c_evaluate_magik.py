#!/usr/bin/env python3
"""
Evaluate MAGIK Tracking Model

Evaluates trained MAGIK model on test data and generates comprehensive metrics.

Usage:
    python scripts/06_evaluate_magik.py \
        --magik checkpoints/magik/best_model.pth \
        --decode checkpoints/decode_optimized/best_model.pth \
        --data data/synthetic_optimized \
        --output results/eval_magik \
        --num_samples 20
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import sys
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import tifffile

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.magik_gnn import MAGIKNet
from piezo1_magik.models.decode_net import DECODENet
from piezo1_magik.data.magik_dataset import MAGIKDataset


def load_models(magik_path, decode_path, device='cpu'):
    """Load MAGIK and DECODE models."""
    print(f"Loading models...")
    
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
    
    print(f"✅ Models loaded:")
    print(f"   DECODE: epoch {decode_checkpoint['epoch']}")
    print(f"   MAGIK: epoch {magik_checkpoint['epoch']}, val loss {magik_checkpoint['val_loss']:.4f}")
    
    return magik_model, decode_model


def greedy_linking(edge_probs, edge_index, threshold=0.5):
    """
    Improved greedy linking algorithm.
    
    Links detections by selecting high-probability edges, ensuring each
    detection is linked at most once in each direction (forward/backward).
    """
    # Get edges above threshold
    mask = edge_probs > threshold
    
    if mask.sum() == 0:
        return {}  # No links
    
    valid_edges = edge_index[:, mask]
    valid_probs = edge_probs[mask]
    
    # Sort by probability (highest first)
    sorted_idx = torch.argsort(valid_probs, descending=True)
    
    # Track linking: each node can have at most one outgoing and one incoming edge
    src_linked = set()  # Nodes that already have outgoing links
    dst_linked = set()  # Nodes that already have incoming links
    
    # Store the actual links
    links = []
    
    for idx in sorted_idx:
        src = valid_edges[0, idx].item()
        dst = valid_edges[1, idx].item()
        
        # Only add link if neither endpoint is already linked in this direction
        if src not in src_linked and dst not in dst_linked:
            links.append((src, dst))
            src_linked.add(src)
            dst_linked.add(dst)
    
    # Build tracks from links
    node_to_track = {}
    track_counter = 0
    
    # Build adjacency structure
    forward_links = {}  # node -> next node
    backward_links = {}  # node -> prev node
    
    for src, dst in links:
        forward_links[src] = dst
        backward_links[dst] = src
    
    # Find all track starts (nodes with no predecessor)
    all_nodes = set(forward_links.keys()) | set(backward_links.keys())
    track_starts = [n for n in all_nodes if n not in backward_links]
    
    # Build tracks by following links forward
    for start_node in track_starts:
        if start_node in node_to_track:
            continue  # Already assigned
        
        # Follow this track
        current = start_node
        while current is not None:
            node_to_track[current] = track_counter
            current = forward_links.get(current, None)
        
        track_counter += 1
    
    # Assign any unlinked nodes to their own tracks
    for node in all_nodes:
        if node not in node_to_track:
            node_to_track[node] = track_counter
            track_counter += 1
    
    return node_to_track


def compute_track_metrics(pred_tracks, gt_tracks, node_features):
    """
    Compute track-level metrics.
    
    Args:
        pred_tracks: Dict mapping node_idx -> predicted_track_id
        gt_tracks: Tensor of ground truth track IDs (N,)
        node_features: Tensor of node features (N, 5)
        
    Returns:
        metrics: Dict with completeness, purity, etc.
    """
    # Build predicted tracks
    pred_track_to_nodes = defaultdict(list)
    for node_idx, track_id in pred_tracks.items():
        pred_track_to_nodes[track_id].append(node_idx)
    
    # Build ground truth tracks
    gt_track_to_nodes = defaultdict(list)
    for node_idx in range(len(gt_tracks)):
        track_id = gt_tracks[node_idx].item()
        gt_track_to_nodes[track_id].append(node_idx)
    
    # Compute metrics for each GT track
    completeness_scores = []
    purity_scores = []
    
    for gt_id, gt_nodes in gt_track_to_nodes.items():
        # Find which predicted tracks contain these nodes
        pred_track_counts = defaultdict(int)
        for node_idx in gt_nodes:
            if node_idx in pred_tracks:
                pred_id = pred_tracks[node_idx]
                pred_track_counts[pred_id] += 1
        
        if len(pred_track_counts) == 0:
            # No predicted track for this GT track
            completeness_scores.append(0.0)
            continue
        
        # Completeness: fraction of GT track that's in best matching pred track
        best_pred_id = max(pred_track_counts, key=pred_track_counts.get)
        completeness = pred_track_counts[best_pred_id] / len(gt_nodes)
        completeness_scores.append(completeness)
        
        # Purity: fraction of matched pred track that's from this GT track
        pred_nodes = pred_track_to_nodes[best_pred_id]
        purity = pred_track_counts[best_pred_id] / len(pred_nodes)
        purity_scores.append(purity)
    
    # Count ID switches
    id_switches = 0
    for gt_id, gt_nodes in gt_track_to_nodes.items():
        # Sort by frame
        gt_nodes_sorted = sorted(gt_nodes, key=lambda n: node_features[n, 0].item())
        
        # Count changes in predicted track ID
        prev_pred_id = None
        for node_idx in gt_nodes_sorted:
            if node_idx in pred_tracks:
                pred_id = pred_tracks[node_idx]
                if prev_pred_id is not None and pred_id != prev_pred_id:
                    id_switches += 1
                prev_pred_id = pred_id
    
    return {
        'completeness': np.mean(completeness_scores) if completeness_scores else 0,
        'purity': np.mean(purity_scores) if purity_scores else 0,
        'num_gt_tracks': len(gt_track_to_nodes),
        'num_pred_tracks': len(pred_track_to_nodes),
        'id_switches': id_switches
    }


def evaluate_sample(magik_model, sample, device, threshold=0.5):
    """Evaluate on a single sample."""
    # Move to device
    node_features = sample['node_features'].to(device)
    edge_index = sample['edge_index'].to(device)
    edge_features = sample['edge_features'].to(device)
    edge_labels = sample['edge_labels'].to(device)
    node_tracks = sample['node_tracks'].to(device)
    
    # Skip if no edges
    if edge_index.shape[1] == 0:
        return None
    
    # Run model
    with torch.no_grad():
        edge_logits = magik_model(node_features, edge_index, edge_features)
        edge_probs = torch.sigmoid(edge_logits)
    
    # Compute edge-level metrics
    pred_labels = (edge_probs > threshold).float()
    
    correct = (pred_labels == edge_labels).sum().item()
    total = edge_labels.numel()
    accuracy = correct / total if total > 0 else 0
    
    tp = ((pred_labels == 1) & (edge_labels == 1)).sum().item()
    fp = ((pred_labels == 1) & (edge_labels == 0)).sum().item()
    fn = ((pred_labels == 0) & (edge_labels == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    edge_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
    
    # Compute track-level metrics
    pred_tracks = greedy_linking(edge_probs.cpu(), edge_index.cpu(), threshold)
    track_metrics = compute_track_metrics(pred_tracks, node_tracks.cpu(), node_features.cpu())
    
    return {
        'edge_metrics': edge_metrics,
        'track_metrics': track_metrics,
        'edge_probs': edge_probs.cpu().numpy(),
        'edge_labels': edge_labels.cpu().numpy()
    }


def evaluate_model(magik_model, decode_model, data_dir, device, 
                  num_samples=20, threshold=0.5, max_temporal_gap=2, 
                  max_spatial_distance=30):
    """Evaluate MAGIK model on test set."""
    
    print(f"\nCreating test dataset...")
    dataset = MAGIKDataset(
        data_dir,
        decode_model,
        device=device,
        max_temporal_gap=max_temporal_gap,
        max_spatial_distance=max_spatial_distance
    )
    
    # Use subset
    num_samples = min(num_samples, len(dataset))
    
    print(f"Evaluating on {num_samples} samples...\n")
    
    results = {
        'edge_metrics': [],
        'track_metrics': [],
        'all_probs': [],
        'all_labels': []
    }
    
    for i in tqdm(range(num_samples), desc='Evaluating'):
        sample = dataset[i]
        
        result = evaluate_sample(magik_model, sample, device, threshold)
        
        if result is not None:
            results['edge_metrics'].append(result['edge_metrics'])
            results['track_metrics'].append(result['track_metrics'])
            results['all_probs'].extend(result['edge_probs'])
            results['all_labels'].extend(result['edge_labels'])
    
    # Aggregate metrics
    edge_metrics = pd.DataFrame(results['edge_metrics'])
    track_metrics = pd.DataFrame(results['track_metrics'])
    
    summary = {
        'edge': {
            'accuracy': float(edge_metrics['accuracy'].mean()),
            'precision': float(edge_metrics['precision'].mean()),
            'recall': float(edge_metrics['recall'].mean()),
            'tp': int(edge_metrics['tp'].sum()),
            'fp': int(edge_metrics['fp'].sum()),
            'fn': int(edge_metrics['fn'].sum())
        },
        'track': {
            'completeness': float(track_metrics['completeness'].mean()),
            'purity': float(track_metrics['purity'].mean()),
            'id_switches': int(track_metrics['id_switches'].sum()),
            'avg_gt_tracks': float(track_metrics['num_gt_tracks'].mean()),
            'avg_pred_tracks': float(track_metrics['num_pred_tracks'].mean())
        },
        'threshold': threshold
    }
    
    results['summary'] = summary
    results['edge_metrics_df'] = edge_metrics
    results['track_metrics_df'] = track_metrics
    
    return results


def plot_results(results, output_dir):
    """Generate evaluation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    
    # 1. Edge Classification Performance
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    edge = results['summary']['edge']
    metrics = ['Accuracy', 'Precision', 'Recall']
    values = [edge['accuracy'], edge['precision'], edge['recall']]
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    axes[0].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Edge Classification Performance', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, (m, v) in enumerate(zip(metrics, values)):
        axes[0].text(i, v + 0.05, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Confusion matrix
    confusion_data = np.array([[edge['tp'], edge['fp']], [edge['fn'], 0]])
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Link', 'Pred No Link'],
                yticklabels=['True Link', 'True No Link'],
                ax=axes[1], cbar=False)
    axes[1].set_title('Edge Classification', fontsize=14, fontweight='bold')
    
    # Summary
    summary_text = f"""
    Edge Classification
    ━━━━━━━━━━━━━━━━━━━━
    Accuracy:  {edge['accuracy']:.3f}
    Precision: {edge['precision']:.3f}
    Recall:    {edge['recall']:.3f}
    
    True Positives:  {edge['tp']}
    False Positives: {edge['fp']}
    False Negatives: {edge['fn']}
    """
    axes[2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'edge_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Track Quality Metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    track = results['summary']['track']
    
    # Completeness and Purity
    metrics = ['Completeness', 'Purity']
    values = [track['completeness'], track['purity']]
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Track Quality', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, (m, v) in enumerate(zip(metrics, values)):
        axes[0].text(i, v + 0.05, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Track counts
    track_df = results['track_metrics_df']
    axes[1].hist(track_df['num_gt_tracks'], bins=20, alpha=0.5, 
                label='Ground Truth', color='gray', edgecolor='black')
    axes[1].hist(track_df['num_pred_tracks'], bins=20, alpha=0.7,
                label='Predicted', color='#3498db', edgecolor='black')
    axes[1].set_xlabel('Number of Tracks', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Track Count Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Summary
    summary_text = f"""
    Track Quality
    ━━━━━━━━━━━━━━━━━━━━
    Completeness: {track['completeness']:.3f}
    Purity:       {track['purity']:.3f}
    
    ID Switches: {track['id_switches']}
    
    Avg GT Tracks:   {track['avg_gt_tracks']:.1f}
    Avg Pred Tracks: {track['avg_pred_tracks']:.1f}
    """
    axes[2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'track_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-sample performance
    edge_df = results['edge_metrics_df']
    track_df = results['track_metrics_df']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Edge metrics over samples
    sample_idx = range(len(edge_df))
    axes[0, 0].plot(sample_idx, edge_df['accuracy'], 'o-', label='Accuracy', color='#3498db')
    axes[0, 0].plot(sample_idx, edge_df['precision'], 's-', label='Precision', color='#2ecc71')
    axes[0, 0].plot(sample_idx, edge_df['recall'], '^-', label='Recall', color='#9b59b6')
    axes[0, 0].set_xlabel('Sample Index', fontsize=10)
    axes[0, 0].set_ylabel('Score', fontsize=10)
    axes[0, 0].set_title('Edge Metrics per Sample', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Track metrics over samples
    axes[0, 1].plot(sample_idx, track_df['completeness'], 'o-', 
                   label='Completeness', color='#2ecc71')
    axes[0, 1].plot(sample_idx, track_df['purity'], 's-',
                   label='Purity', color='#e74c3c')
    axes[0, 1].set_xlabel('Sample Index', fontsize=10)
    axes[0, 1].set_ylabel('Score', fontsize=10)
    axes[0, 1].set_title('Track Quality per Sample', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Completeness distribution
    axes[1, 0].hist(track_df['completeness'], bins=20, edgecolor='black',
                   alpha=0.7, color='#2ecc71')
    axes[1, 0].axvline(track_df['completeness'].mean(), color='red',
                      linestyle='--', linewidth=2,
                      label=f"Mean: {track_df['completeness'].mean():.3f}")
    axes[1, 0].set_xlabel('Completeness', fontsize=10)
    axes[1, 0].set_ylabel('Count', fontsize=10)
    axes[1, 0].set_title('Track Completeness Distribution', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Purity distribution
    axes[1, 1].hist(track_df['purity'], bins=20, edgecolor='black',
                   alpha=0.7, color='#e74c3c')
    axes[1, 1].axvline(track_df['purity'].mean(), color='blue',
                      linestyle='--', linewidth=2,
                      label=f"Mean: {track_df['purity'].mean():.3f}")
    axes[1, 1].set_xlabel('Purity', fontsize=10)
    axes[1, 1].set_ylabel('Count', fontsize=10)
    axes[1, 1].set_title('Track Purity Distribution', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_sample_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plots saved to: {output_dir}")


def plot_visual_inspection(magik_model, decode_model, data_dir, device, 
                          output_dir, threshold=0.5, num_samples=6,
                          max_temporal_gap=2, max_spatial_distance=30):
    """
    Create visual inspection plots showing tracks overlaid on images.
    """
    output_dir = Path(output_dir)
    
    print(f"\nCreating visual inspection plots...")
    
    # Load a few samples
    dataset = MAGIKDataset(
        data_dir, decode_model, device,
        max_temporal_gap=max_temporal_gap,
        max_spatial_distance=max_spatial_distance
    )
    
    # Select samples
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    sample_indices = np.linspace(0, total_samples - 1, num_samples, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for plot_idx, sample_idx in enumerate(sample_indices):
        if plot_idx >= len(axes):
            break
        
        # Get sample
        sample_dir = dataset.samples[sample_idx]
        
        # Load movie
        movie_path = sample_dir / 'movie.tif'
        movie = tifffile.imread(movie_path)
        
        # Load ground truth
        gt_path = sample_dir / 'ground_truth_tracks.csv'
        gt_df = pd.read_csv(gt_path)
        
        # Get middle frame for display
        middle_frame = movie.shape[0] // 2
        frame_img = movie[middle_frame]
        
        # Get graph data
        graph_sample = dataset[sample_idx]
        node_features = graph_sample['node_features'].to(device)
        edge_index = graph_sample['edge_index'].to(device)
        edge_features = graph_sample['edge_features'].to(device)
        node_tracks_gt = graph_sample['node_tracks']
        
        # Run MAGIK
        with torch.no_grad():
            edge_logits = magik_model(node_features, edge_index, edge_features)
            edge_probs = torch.sigmoid(edge_logits)
        
        # Get predicted tracks
        pred_tracks = greedy_linking(edge_probs.cpu(), edge_index.cpu(), threshold)
        
        # Plot image
        ax = axes[plot_idx]
        vmin, vmax = np.percentile(frame_img, [1, 99])
        ax.imshow(frame_img, cmap='gray', vmin=vmin, vmax=vmax)
        
        # Get detections in middle frame (±2 frames for visibility)
        frame_range = range(max(0, middle_frame - 2), min(movie.shape[0], middle_frame + 3))
        
        # Plot predicted tracks
        track_colors = {}
        for node_idx, track_id in pred_tracks.items():
            frame = node_features[node_idx, 0].item()
            if frame not in frame_range:
                continue
            
            x = node_features[node_idx, 1].item()
            y = node_features[node_idx, 2].item()
            
            # Get color for this track
            if track_id not in track_colors:
                track_colors[track_id] = plt.cm.tab20(len(track_colors) % 20)
            
            color = track_colors[track_id]
            ax.scatter([x], [y], s=80, facecolors='none', 
                      edgecolors=color, linewidths=2, alpha=0.8)
        
        # Draw links between consecutive frames
        for node_idx, track_id in pred_tracks.items():
            frame = node_features[node_idx, 0].item()
            if frame not in frame_range[:-1]:  # Not last frame
                continue
            
            # Find next node in track
            x1 = node_features[node_idx, 1].item()
            y1 = node_features[node_idx, 2].item()
            
            for other_idx, other_track_id in pred_tracks.items():
                if other_track_id != track_id:
                    continue
                
                other_frame = node_features[other_idx, 0].item()
                if other_frame == frame + 1:  # Next frame
                    x2 = node_features[other_idx, 1].item()
                    y2 = node_features[other_idx, 2].item()
                    
                    color = track_colors[track_id]
                    ax.plot([x1, x2], [y1, y2], '-', color=color, 
                           linewidth=1.5, alpha=0.6)
                    break
        
        # Metrics
        result = evaluate_sample(magik_model, graph_sample, device, threshold)
        if result is not None:
            edge_m = result['edge_metrics']
            track_m = result['track_metrics']
            
            ax.set_title(f'Sample {sample_idx} (Frame {middle_frame})\n' +
                        f'Edge: P={edge_m["precision"]:.2f}, R={edge_m["recall"]:.2f} | ' +
                        f'Track: C={track_m["completeness"]:.2f}, P={track_m["purity"]:.2f}',
                        fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'Sample {sample_idx} (Frame {middle_frame})',
                        fontsize=10, fontweight='bold')
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(sample_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Visual Inspection: Predicted Tracks Overlaid on Images\n' +
                'Each color = one predicted track, Lines show links between frames',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'visual_inspection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visual inspection saved: {output_dir / 'visual_inspection.png'}")
    
    # Also create a detailed track comparison plot
    create_track_comparison(magik_model, decode_model, data_dir, device, 
                           output_dir, threshold, max_temporal_gap, max_spatial_distance)


def create_track_comparison(magik_model, decode_model, data_dir, device,
                            output_dir, threshold=0.5, max_temporal_gap=2, 
                            max_spatial_distance=30):
    """Create detailed comparison of GT vs predicted tracks."""
    
    dataset = MAGIKDataset(
        data_dir, decode_model, device,
        max_temporal_gap=max_temporal_gap,
        max_spatial_distance=max_spatial_distance
    )
    
    # Pick one good sample
    sample_idx = len(dataset) // 2
    sample_dir = dataset.samples[sample_idx]
    
    # Load movie and GT
    movie_path = sample_dir / 'movie.tif'
    movie = tifffile.imread(movie_path)
    
    gt_path = sample_dir / 'ground_truth_tracks.csv'
    gt_df = pd.read_csv(gt_path)
    
    # Get predictions
    graph_sample = dataset[sample_idx]
    node_features = graph_sample['node_features'].to(device)
    edge_index = graph_sample['edge_index'].to(device)
    edge_features = graph_sample['edge_features'].to(device)
    
    with torch.no_grad():
        edge_logits = magik_model(node_features, edge_index, edge_features)
        edge_probs = torch.sigmoid(edge_logits)
    
    pred_tracks = greedy_linking(edge_probs.cpu(), edge_index.cpu(), threshold)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pick a middle frame
    middle_frame = movie.shape[0] // 2
    frame_img = movie[middle_frame]
    vmin, vmax = np.percentile(frame_img, [1, 99])
    
    # Left: Ground truth tracks
    ax = axes[0]
    ax.imshow(frame_img, cmap='gray', vmin=vmin, vmax=vmax)
    
    # Plot GT tracks
    frame_range = range(max(0, middle_frame - 5), min(movie.shape[0], middle_frame + 6))
    
    for track_id in gt_df['track_id'].unique():
        track_df = gt_df[gt_df['track_id'] == track_id]
        track_df = track_df[track_df['frame'].isin(frame_range)]
        
        if len(track_df) == 0:
            continue
        
        color = plt.cm.tab20(track_id % 20)
        
        # Plot points
        ax.scatter(track_df['x'], track_df['y'], s=80, 
                  facecolors='none', edgecolors=color, linewidths=2, alpha=0.8)
        
        # Plot trajectory
        track_df_sorted = track_df.sort_values('frame')
        ax.plot(track_df_sorted['x'], track_df_sorted['y'], '-', 
               color=color, linewidth=1.5, alpha=0.6)
    
    ax.set_title(f'Ground Truth Tracks (Frame {middle_frame}±5)\n' +
                f'{len(gt_df["track_id"].unique())} total tracks',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Right: Predicted tracks
    ax = axes[1]
    ax.imshow(frame_img, cmap='gray', vmin=vmin, vmax=vmax)
    
    # Build predicted track trajectories
    pred_track_to_nodes = {}
    for node_idx, track_id in pred_tracks.items():
        if track_id not in pred_track_to_nodes:
            pred_track_to_nodes[track_id] = []
        pred_track_to_nodes[track_id].append(node_idx)
    
    # Plot predicted tracks
    track_colors = {}
    for track_id, node_indices in pred_track_to_nodes.items():
        # Get positions
        positions = []
        for node_idx in node_indices:
            frame = node_features[node_idx, 0].item()
            if frame not in frame_range:
                continue
            
            x = node_features[node_idx, 1].item()
            y = node_features[node_idx, 2].item()
            positions.append((frame, x, y))
        
        if len(positions) == 0:
            continue
        
        # Color
        if track_id not in track_colors:
            track_colors[track_id] = plt.cm.tab20(len(track_colors) % 20)
        color = track_colors[track_id]
        
        # Plot
        positions = sorted(positions, key=lambda p: p[0])
        frames, xs, ys = zip(*positions)
        
        ax.scatter(xs, ys, s=80, facecolors='none', 
                  edgecolors=color, linewidths=2, alpha=0.8)
        ax.plot(xs, ys, '-', color=color, linewidth=1.5, alpha=0.6)
    
    ax.set_title(f'Predicted Tracks (Frame {middle_frame}±5)\n' +
                f'{len(pred_track_to_nodes)} total tracks',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle(f'Track Comparison: Sample {sample_idx}\n' +
                'Ground Truth vs MAGIK Predictions',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'track_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Track comparison saved: {output_dir / 'track_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MAGIK tracking model')
    parser.add_argument('--magik', type=str, required=True,
                       help='Path to trained MAGIK model')
    parser.add_argument('--decode', type=str, required=True,
                       help='Path to trained DECODE model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output', type=str, default='results/eval_magik',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to evaluate')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Edge probability threshold')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU ID (-1 for CPU)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MAGIK MODEL EVALUATION")
    print("="*70)
    print(f"MAGIK model: {args.magik}")
    print(f"DECODE model: {args.decode}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Samples: {args.num_samples}")
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
    magik_model, decode_model = load_models(args.magik, args.decode, device)
    
    # Evaluate
    results = evaluate_model(
        magik_model, decode_model, args.data, device,
        num_samples=args.num_samples,
        threshold=args.threshold
    )
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    edge = results['summary']['edge']
    track = results['summary']['track']
    
    print(f"\n📊 Edge Classification:")
    print(f"   Accuracy:  {edge['accuracy']:.3f}")
    print(f"   Precision: {edge['precision']:.3f}")
    print(f"   Recall:    {edge['recall']:.3f}")
    
    print(f"\n📈 Edge Counts:")
    print(f"   True Positives:  {edge['tp']}")
    print(f"   False Positives: {edge['fp']}")
    print(f"   False Negatives: {edge['fn']}")
    
    print(f"\n🎯 Track Quality:")
    print(f"   Completeness: {track['completeness']:.3f}")
    print(f"   Purity:       {track['purity']:.3f}")
    print(f"   ID Switches:  {track['id_switches']}")
    
    print(f"\n📊 Track Counts:")
    print(f"   Avg GT Tracks:   {track['avg_gt_tracks']:.1f}")
    print(f"   Avg Pred Tracks: {track['avg_pred_tracks']:.1f}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(results['summary'], f, indent=2)
    
    # Save detailed CSVs
    results['edge_metrics_df'].to_csv(output_dir / 'edge_metrics.csv', index=False)
    results['track_metrics_df'].to_csv(output_dir / 'track_metrics.csv', index=False)
    
    # Generate plots
    plot_results(results, output_dir)
    
    # Generate visual inspection plots
    plot_visual_inspection(
        magik_model, decode_model, args.data, device, output_dir,
        threshold=args.threshold, num_samples=6
    )
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - evaluation_summary.json")
    print(f"  - edge_metrics.csv")
    print(f"  - track_metrics.csv")
    print(f"  - edge_performance.png")
    print(f"  - track_quality.png")
    print(f"  - per_sample_analysis.png")
    print(f"  - visual_inspection.png (NEW!)")
    print(f"  - track_comparison.png (NEW!)")


if __name__ == '__main__':
    main()
