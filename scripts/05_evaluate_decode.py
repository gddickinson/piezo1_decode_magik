#!/usr/bin/env python3
"""
Evaluate Trained DECODE Model

Evaluates localization accuracy, detection performance, and generates
comprehensive visualizations and metrics.

Usage:
    python 05_evaluate_decode.py \
        --model checkpoints/decode_test/best_model.pth \
        --data data/synthetic_test \
        --output results/evaluation \
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from piezo1_magik.models.decode_net import DECODENet
from piezo1_magik.data.decode_dataset import DECODEDataset
import tifffile


def load_model(checkpoint_path, device='cpu'):
    """Load trained DECODE model."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config
    config = checkpoint.get('config', {})
    base_channels = config.get('model', {}).get('base_channels', 32)
    
    # Create and load model
    model = DECODENet(base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']}, val loss: {checkpoint.get('best_val_loss', 'N/A')})")
    
    return model


def predict_puncta(model, images, device='cpu', threshold=0.5):
    """
    Run DECODE inference on 3-frame window.
    
    Args:
        model: Trained DECODE model
        images: (3, H, W) numpy array (already normalized to [0, 1])
        device: Device to run on
        threshold: Detection probability threshold
        
    Returns:
        detections: List of dicts with {x, y, prob, photons}
    """
    # Convert to tensor
    images_tensor = torch.from_numpy(images).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # Predict
    with torch.no_grad():
        outputs = model(images_tensor)
    
    # Get predictions
    prob = outputs['prob'][0, 0].cpu().numpy()  # (H, W)
    offset = outputs['offset'][0].cpu().numpy()  # (2, H, W)
    photons = outputs['photons'][0, 0].cpu().numpy()  # (H, W)
    
    # Find detections above threshold
    detections = []
    detection_mask = prob > threshold
    
    y_indices, x_indices = np.where(detection_mask)
    
    for y, x in zip(y_indices, x_indices):
        detections.append({
            'x': x + offset[0, y, x],  # Sub-pixel x
            'y': y + offset[1, y, x],  # Sub-pixel y
            'prob': prob[y, x],
            'photons': photons[y, x]
        })
    
    return detections


def match_detections(pred_detections, gt_detections, max_distance=3.0):
    """
    Match predicted detections to ground truth.
    
    Args:
        pred_detections: List of predicted detections
        gt_detections: List of ground truth detections
        max_distance: Maximum distance (pixels) for matching
        
    Returns:
        matches: List of (pred_idx, gt_idx, distance)
        tp_indices: True positive prediction indices
        fp_indices: False positive prediction indices
        fn_indices: False negative ground truth indices
    """
    if len(pred_detections) == 0:
        return [], [], [], list(range(len(gt_detections)))
    
    if len(gt_detections) == 0:
        return [], [], list(range(len(pred_detections))), []
    
    # Compute distance matrix
    pred_coords = np.array([[d['x'], d['y']] for d in pred_detections])
    gt_coords = np.array([[d['x'], d['y']] for d in gt_detections])
    
    # Distance matrix (n_pred, n_gt)
    distances = np.sqrt(np.sum((pred_coords[:, None] - gt_coords[None, :])**2, axis=2))
    
    # Greedy matching
    matches = []
    matched_pred = set()
    matched_gt = set()
    
    while True:
        # Find minimum distance
        min_dist = distances.min()
        
        if min_dist > max_distance:
            break
        
        # Find indices
        pred_idx, gt_idx = np.unravel_index(distances.argmin(), distances.shape)
        
        if pred_idx in matched_pred or gt_idx in matched_gt:
            distances[pred_idx, gt_idx] = np.inf
            continue
        
        matches.append((int(pred_idx), int(gt_idx), float(min_dist)))
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        distances[pred_idx, gt_idx] = np.inf
    
    # Classify predictions
    tp_indices = [m[0] for m in matches]
    fp_indices = [i for i in range(len(pred_detections)) if i not in matched_pred]
    fn_indices = [i for i in range(len(gt_detections)) if i not in matched_gt]
    
    return matches, tp_indices, fp_indices, fn_indices


def evaluate_model(model, data_dir, device='cpu', num_samples=20, threshold=0.5):
    """
    Evaluate model on dataset.
    
    Returns:
        results: Dict with metrics and per-sample results
    """
    print(f"\nEvaluating on {num_samples} samples...")
    
    # Load dataset
    dataset = DECODEDataset(data_dir)
    
    # Limit samples
    num_samples = min(num_samples, len(dataset))
    
    # Storage for results
    all_matches = []
    all_localization_errors = []
    all_photon_errors = []
    
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    per_sample_results = []
    
    for idx in tqdm(range(num_samples), desc='Evaluating'):
        sample = dataset[idx]
        
        # Get images and ground truth
        images = sample['images'].numpy()  # (3, H, W)
        has_puncta = sample['has_puncta'].numpy()[0]  # (H, W)
        offset = sample['offset'].numpy()  # (2, H, W)
        photons_gt = sample['photons'].numpy()[0]  # (H, W)
        
        # Find ground truth detections
        gt_y, gt_x = np.where(has_puncta > 0.5)
        gt_detections = []
        for y, x in zip(gt_y, gt_x):
            gt_detections.append({
                'x': x + offset[0, y, x],
                'y': y + offset[1, y, x],
                'photons': photons_gt[y, x]
            })
        
        # Predict
        pred_detections = predict_puncta(model, images, device, threshold)
        
        # Match detections
        matches, tp_idx, fp_idx, fn_idx = match_detections(
            pred_detections, gt_detections, max_distance=3.0
        )
        
        # Update counts
        tp_count += len(matches)
        fp_count += len(fp_idx)
        fn_count += len(fn_idx)
        
        # Compute localization errors for matches
        for pred_idx, gt_idx, dist in matches:
            pred = pred_detections[pred_idx]
            gt = gt_detections[gt_idx]
            
            all_localization_errors.append(dist)
            all_matches.append({
                'sample_idx': idx,
                'pred_x': pred['x'],
                'pred_y': pred['y'],
                'gt_x': gt['x'],
                'gt_y': gt['y'],
                'distance': dist,
                'pred_photons': pred['photons'],
                'gt_photons': gt['photons']
            })
            
            # Photon error
            photon_error = abs(pred['photons'] - gt['photons'])
            all_photon_errors.append(photon_error)
        
        # Store per-sample results
        per_sample_results.append({
            'sample_idx': idx,
            'n_gt': len(gt_detections),
            'n_pred': len(pred_detections),
            'tp': len(matches),
            'fp': len(fp_idx),
            'fn': len(fn_idx),
            'precision': len(matches) / len(pred_detections) if len(pred_detections) > 0 else 0,
            'recall': len(matches) / len(gt_detections) if len(gt_detections) > 0 else 0
        })
    
    # Compute overall metrics
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Localization metrics (convert to nm, assuming 130 nm pixel size)
    pixel_size_nm = 130.0
    localization_errors_nm = np.array(all_localization_errors) * pixel_size_nm
    
    rmse = np.sqrt(np.mean(localization_errors_nm**2)) if len(localization_errors_nm) > 0 else 0
    mean_error = np.mean(localization_errors_nm) if len(localization_errors_nm) > 0 else 0
    median_error = np.median(localization_errors_nm) if len(localization_errors_nm) > 0 else 0
    
    # Photon metrics
    mean_photon_error = np.mean(all_photon_errors) if len(all_photon_errors) > 0 else 0
    
    results = {
        'detection': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp_count,
            'fp': fp_count,
            'fn': fn_count
        },
        'localization': {
            'rmse_nm': rmse,
            'mean_error_nm': mean_error,
            'median_error_nm': median_error,
            'errors_nm': localization_errors_nm.tolist()
        },
        'photons': {
            'mean_error': mean_photon_error,
            'errors': all_photon_errors
        },
        'matches': all_matches,
        'per_sample': per_sample_results
    }
    
    return results


def plot_results(results, output_dir):
    """Generate evaluation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    
    # 1. Detection performance
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    det = results['detection']
    metrics = ['Precision', 'Recall', 'F1']
    values = [det['precision'], det['recall'], det['f1']]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    axes[0].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Detection Performance', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, (m, v) in enumerate(zip(metrics, values)):
        axes[0].text(i, v + 0.05, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Confusion matrix style
    confusion_data = np.array([[det['tp'], det['fp']], [det['fn'], 0]])
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'],
                ax=axes[1], cbar=False)
    axes[1].set_title('Detection Counts', fontsize=14, fontweight='bold')
    
    # Summary text
    summary_text = f"""
    Detection Performance
    ━━━━━━━━━━━━━━━━━━━━
    Precision: {det['precision']:.3f}
    Recall:    {det['recall']:.3f}
    F1 Score:  {det['f1']:.3f}
    
    True Positives:  {det['tp']}
    False Positives: {det['fp']}
    False Negatives: {det['fn']}
    """
    axes[2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Localization error distribution
    errors_nm = results['localization']['errors_nm']
    
    if len(errors_nm) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(errors_nm, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        axes[0].axvline(results['localization']['mean_error_nm'], 
                       color='red', linestyle='--', linewidth=2, 
                       label=f"Mean: {results['localization']['mean_error_nm']:.1f} nm")
        axes[0].axvline(results['localization']['median_error_nm'], 
                       color='orange', linestyle='--', linewidth=2,
                       label=f"Median: {results['localization']['median_error_nm']:.1f} nm")
        axes[0].set_xlabel('Localization Error (nm)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Localization Error Distribution', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_errors = np.sort(errors_nm)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        axes[1].plot(sorted_errors, cumulative * 100, linewidth=2, color='#3498db')
        axes[1].axhline(50, color='orange', linestyle='--', alpha=0.5, 
                       label=f"50%: {np.percentile(errors_nm, 50):.1f} nm")
        axes[1].axhline(90, color='red', linestyle='--', alpha=0.5,
                       label=f"90%: {np.percentile(errors_nm, 90):.1f} nm")
        axes[1].set_xlabel('Localization Error (nm)', fontsize=12)
        axes[1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
        axes[1].set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'localization_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Per-sample performance
    per_sample = pd.DataFrame(results['per_sample'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(per_sample['sample_idx'], per_sample['precision'], 'o-', 
                    label='Precision', color='#2ecc71')
    axes[0, 0].plot(per_sample['sample_idx'], per_sample['recall'], 's-', 
                    label='Recall', color='#3498db')
    axes[0, 0].set_xlabel('Sample Index', fontsize=10)
    axes[0, 0].set_ylabel('Score', fontsize=10)
    axes[0, 0].set_title('Per-Sample Detection Performance', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(per_sample['sample_idx'], per_sample['n_gt'], 
                   alpha=0.5, label='Ground Truth', color='gray')
    axes[0, 1].bar(per_sample['sample_idx'], per_sample['n_pred'], 
                   alpha=0.7, label='Predicted', color='#3498db')
    axes[0, 1].set_xlabel('Sample Index', fontsize=10)
    axes[0, 1].set_ylabel('Count', fontsize=10)
    axes[0, 1].set_title('Detections per Sample', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].bar(per_sample['sample_idx'], per_sample['tp'], 
                   label='TP', color='#2ecc71', alpha=0.7)
    axes[1, 0].bar(per_sample['sample_idx'], per_sample['fp'], 
                   bottom=per_sample['tp'], label='FP', color='#e74c3c', alpha=0.7)
    axes[1, 0].set_xlabel('Sample Index', fontsize=10)
    axes[1, 0].set_ylabel('Count', fontsize=10)
    axes[1, 0].set_title('True/False Positives per Sample', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Summary statistics
    stats_text = f"""
    Overall Statistics
    ━━━━━━━━━━━━━━━━━━━━━━━━
    Samples Evaluated: {len(per_sample)}
    
    Localization RMSE: {results['localization']['rmse_nm']:.1f} nm
    Mean Error: {results['localization']['mean_error_nm']:.1f} nm
    Median Error: {results['localization']['median_error_nm']:.1f} nm
    
    Avg Precision: {per_sample['precision'].mean():.3f}
    Avg Recall: {per_sample['recall'].mean():.3f}
    
    Total GT Puncta: {per_sample['n_gt'].sum()}
    Total Predictions: {per_sample['n_pred'].sum()}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_sample_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DECODE model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory with synthetic samples')
    parser.add_argument('--output', type=str, default='results/evaluation',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to evaluate')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection probability threshold')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (-1 for CPU)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DECODE MODEL EVALUATION")
    print("="*70)
    print(f"Model: {args.model}")
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
    
    # Load model
    model = load_model(args.model, device)
    
    # Evaluate
    results = evaluate_model(
        model, args.data, device, 
        num_samples=args.num_samples,
        threshold=args.threshold
    )
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\n📊 Detection Performance:")
    print(f"   Precision: {results['detection']['precision']:.3f}")
    print(f"   Recall:    {results['detection']['recall']:.3f}")
    print(f"   F1 Score:  {results['detection']['f1']:.3f}")
    print(f"\n📍 Localization Performance:")
    print(f"   RMSE:      {results['localization']['rmse_nm']:.1f} nm")
    print(f"   Mean:      {results['localization']['mean_error_nm']:.1f} nm")
    print(f"   Median:    {results['localization']['median_error_nm']:.1f} nm")
    print(f"\n💡 Photon Estimation:")
    print(f"   Mean Error: {results['photons']['mean_error']:.1f} photons")
    print(f"\n📈 Counts:")
    print(f"   True Positives:  {results['detection']['tp']}")
    print(f"   False Positives: {results['detection']['fp']}")
    print(f"   False Negatives: {results['detection']['fn']}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = {
        'model_path': args.model,
        'data_path': args.data,
        'num_samples': args.num_samples,
        'threshold': args.threshold,
        'detection': results['detection'],
        'localization': {
            'rmse_nm': results['localization']['rmse_nm'],
            'mean_error_nm': results['localization']['mean_error_nm'],
            'median_error_nm': results['localization']['median_error_nm']
        },
        'photons': {
            'mean_error': results['photons']['mean_error']
        }
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    matches_df = pd.DataFrame(results['matches'])
    matches_df.to_csv(output_dir / 'matches.csv', index=False)
    
    per_sample_df = pd.DataFrame(results['per_sample'])
    per_sample_df.to_csv(output_dir / 'per_sample_results.csv', index=False)
    
    # Generate plots
    plot_results(results, output_dir)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - evaluation_summary.json")
    print(f"  - matches.csv")
    print(f"  - per_sample_results.csv")
    print(f"  - detection_performance.png")
    print(f"  - localization_errors.png")
    print(f"  - per_sample_analysis.png")


if __name__ == '__main__':
    main()
