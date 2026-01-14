#!/usr/bin/env python3
"""
Resume DECODE Training from Checkpoint

Simple script to resume training after interruption.

Usage:
    # Resume from latest checkpoint
    python scripts/02_resume_training.py \\
        --checkpoint checkpoints/decode_test/latest.pth \\
        --config configs/decode_training.yaml \\
        --data data/synthetic_test \\
        --output checkpoints/decode_test
    
    # Resume from specific checkpoint
    python scripts/02_resume_training.py \\
        --checkpoint checkpoints/decode_test/checkpoint_epoch_20.pth \\
        --config configs/decode_training.yaml \\
        --data data/synthetic_test \\
        --output checkpoints/decode_test
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Resume DECODE training')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file to resume from (e.g., latest.pth)')
    parser.add_argument('--config', type=str, required=True,
                        help='Config file')
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print(f"\nAvailable checkpoints in {args.output}:")
        output_dir = Path(args.output)
        if output_dir.exists():
            for ckpt in sorted(output_dir.glob('*.pth')):
                print(f"   - {ckpt.name}")
        sys.exit(1)
    
    print("="*70)
    print("RESUMING DECODE TRAINING")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print("="*70)
    print()
    
    # Call main training script with --resume flag
    cmd = [
        'python', 'scripts/02_train_decode.py',
        '--config', args.config,
        '--data', args.data,
        '--output', args.output,
        '--resume', args.checkpoint,
        '--gpu', str(args.gpu)
    ]
    
    # Run the training script
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
