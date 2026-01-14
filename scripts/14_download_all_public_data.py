#!/usr/bin/env python3
"""
Download All Public Datasets

Master script to download all recommended public datasets.

Usage:
    python scripts/14_download_all_public_data.py \
        --output data/public_datasets
"""

import argparse
import subprocess
from pathlib import Path


def run_command(cmd):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print('='*70)
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download all public particle tracking datasets'
    )
    parser.add_argument('--output', type=str,
                       default='data/public_datasets',
                       help='Output directory')
    parser.add_argument('--skip_smlm', action='store_true',
                       help='Skip SMLM Challenge datasets')
    parser.add_argument('--skip_tracking', action='store_true',
                       help='Skip Particle Tracking Challenge')
    parser.add_argument('--skip_zenodo', action='store_true',
                       help='Skip Zenodo datasets')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PUBLIC DATASET DOWNLOADER")
    print("="*70)
    print("\nThis will download several GB of data.")
    print("Estimated time: 10-30 minutes (depending on connection)")
    print("="*70)
    
    scripts_dir = Path(__file__).parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. SMLM Challenge 2016
    if not args.skip_smlm:
        print("\n\n" + "="*70)
        print("STEP 1: SMLM CHALLENGE 2016 DATASETS")
        print("="*70)
        print("Downloading simulated TIRF microscopy data...")
        print("Datasets: Tubules, Vesicles (High/Low density)")
        print("Size: ~500 MB total")
        
        cmd = [
            'python', str(scripts_dir / '11_download_smlm_challenge.py'),
            '--output', str(output_dir / 'smlm_challenge'),
            '--datasets', 'vesicles_low,tubules_low'  # Start with easier ones
        ]
        
        results['smlm_challenge'] = run_command(cmd)
    else:
        print("\n⏭️  Skipping SMLM Challenge datasets")
        results['smlm_challenge'] = 'skipped'
    
    # 2. Particle Tracking Challenge
    if not args.skip_tracking:
        print("\n\n" + "="*70)
        print("STEP 2: PARTICLE TRACKING CHALLENGE")
        print("="*70)
        print("Downloading particle tracking benchmarks...")
        print("Datasets: Receptors, Vesicles, Microtubules")
        print("Size: ~300 MB total")
        
        cmd = [
            'python', str(scripts_dir / '12_download_tracking_challenge.py'),
            '--output', str(output_dir / 'tracking_challenge'),
            '--datasets', 'receptor_2d,vesicles_2d'
        ]
        
        results['tracking_challenge'] = run_command(cmd)
    else:
        print("\n⏭️  Skipping Particle Tracking Challenge datasets")
        results['tracking_challenge'] = 'skipped'
    
    # 3. Zenodo datasets
    if not args.skip_zenodo:
        print("\n\n" + "="*70)
        print("STEP 3: ZENODO RESEARCH DATASETS")
        print("="*70)
        print("Creating download instructions for Zenodo datasets...")
        print("(These require manual download from Zenodo website)")
        
        cmd = [
            'python', str(scripts_dir / '13_download_zenodo_datasets.py'),
            '--output', str(output_dir / 'zenodo')
        ]
        
        results['zenodo'] = run_command(cmd)
    else:
        print("\n⏭️  Skipping Zenodo datasets")
        results['zenodo'] = 'skipped'
    
    # Summary
    print("\n\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    
    for dataset, status in results.items():
        if status == 'skipped':
            print(f"⏭️  {dataset}: Skipped")
        elif status:
            print(f"✅ {dataset}: Success")
        else:
            print(f"❌ {dataset}: Failed")
    
    # Create master README
    create_master_readme(output_dir, results)
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"1. Check downloaded data: {output_dir}")
    print(f"2. Read overview: {output_dir}/README.md")
    print("3. Test on a dataset:")
    print("\n   Recommended first test - SMLM Vesicles (low density):")
    print(f"\n   python scripts/09_run_complete_pipeline.py \\")
    print(f"       --input {output_dir}/smlm_challenge/vesicles_low/sequence.tif \\")
    print(f"       --decode checkpoints/decode_optimized/best_model.pth \\")
    print(f"       --magik checkpoints/magik/best_model.pth \\")
    print(f"       --output results/test_vesicles_low \\")
    print(f"       --use_gap_filling")
    
    print("\n" + "="*70)
    print("For Zenodo datasets:")
    print("="*70)
    print(f"Follow manual download instructions in:")
    print(f"{output_dir}/zenodo/DOWNLOAD_INSTRUCTIONS.md")


def create_master_readme(output_dir, results):
    """Create master README for all datasets."""
    readme_path = output_dir / 'README.md'
    
    content = f"""# Public Particle Tracking Datasets

Downloaded datasets for testing DECODE+MAGIK pipeline.

## Directory Structure

```
{output_dir.name}/
├── smlm_challenge/          # SMLM Challenge 2016
│   ├── vesicles_low/        # Simulated vesicles, low density
│   ├── vesicles_high/       # Simulated vesicles, high density
│   ├── tubules_low/         # Simulated microtubules, low density
│   └── tubules_high/        # Simulated microtubules, high density
├── tracking_challenge/      # Particle Tracking Challenge
│   ├── receptor_2d/         # Receptor diffusion
│   ├── vesicles_2d/         # Vesicle trafficking
│   └── microtubules_2d/     # Microtubule dynamics
└── zenodo/                  # Zenodo research datasets
    ├── nino_vesicles/       # TIRF vesicle exocytosis
    ├── manzo_spt/           # SPT benchmarks
    └── DOWNLOAD_INSTRUCTIONS.md
```

## Dataset Categories

### 1. SMLM Challenge 2016 (Best for Initial Testing)

**Source:** http://bigwww.epfl.ch/smlm/challenge2016/

**Characteristics:**
- Simulated TIRF microscopy
- Ground truth available
- Multiple density levels
- Well-characterized blinking

**Recommended for:**
- Testing DECODE detection accuracy
- Validating against known ground truth
- Benchmarking performance

**Best starting dataset:** `vesicles_low` - easier, good for initial tests

### 2. Particle Tracking Challenge (Best for Tracking Validation)

**Source:** http://particletracking.github.io/

**Characteristics:**
- Simulated particle dynamics
- Various motion patterns
- Ground truth trajectories
- Benchmark datasets

**Recommended for:**
- Testing MAGIK tracking accuracy
- Comparing track quality metrics
- Validating fragmentation levels

**Best starting dataset:** `receptor_2d` - simple diffusion

### 3. Zenodo Datasets (Most Similar to Real Data)

**Source:** Various published research

**Characteristics:**
- Real experimental data
- Published in peer-reviewed papers
- Includes analysis code
- Most realistic

**Recommended for:**
- Final validation before using your data
- Comparing to published methods
- Understanding real-world challenges

**Best starting dataset:** Nino vesicles - most similar to PIEZO1

## Quick Start Guide

### Test 1: Detection Accuracy (SMLM Challenge)

```bash
python scripts/09_run_complete_pipeline.py \\
    --input data/public_datasets/smlm_challenge/vesicles_low/sequence.tif \\
    --decode checkpoints/decode_optimized/best_model.pth \\
    --magik checkpoints/magik/best_model.pth \\
    --output results/test_smlm_vesicles \\
    --use_gap_filling
```

**Expected:** ~15 particles/frame, ~300 frames

### Test 2: Tracking Quality (Tracking Challenge)

```bash
python scripts/09_run_complete_pipeline.py \\
    --input data/public_datasets/tracking_challenge/receptor_2d/01_Receptor.tif \\
    --decode checkpoints/decode_optimized/best_model.pth \\
    --magik checkpoints/magik/best_model.pth \\
    --output results/test_receptors \\
    --use_gap_filling
```

**Expected:** ~20 particles/frame, ~500 frames, diffusive motion

### Test 3: Real Data (Zenodo - after manual download)

```bash
python scripts/09_run_complete_pipeline.py \\
    --input data/public_datasets/zenodo/nino_vesicles/movie.tif \\
    --decode checkpoints/decode_optimized/best_model.pth \\
    --magik checkpoints/magik/best_model.pth \\
    --output results/test_zenodo_nino \\
    --use_gap_filling
```

## Comparing to Ground Truth

### For SMLM Challenge:

```python
import pandas as pd
import numpy as np

# Load ground truth
gt = pd.read_csv('data/public_datasets/smlm_challenge/vesicles_low/groundtruth.csv')

# Load your detections
detections = pd.read_csv('results/test_smlm_vesicles/tracks.csv')

# Compare detection accuracy
# (Match detections to GT within 3 pixels)
```

### For Tracking Challenge:

```python
# Load ground truth tracks
gt_tracks = pd.read_csv('data/public_datasets/tracking_challenge/receptor_2d/01_GT.txt',
                        sep='\\t', names=['id', 'frame', 'x', 'y', 'z'])

# Load your tracks
your_tracks = pd.read_csv('results/test_receptors/tracks.csv')

# Compare tracking metrics
# (Fragmentation, completeness, purity)
```

## Performance Expectations

### DECODE (Detection):

- **SMLM Challenge:** F1 > 95% (models trained on similar synthetic data)
- **Tracking Challenge:** F1 > 90% (slightly different PSF)
- **Zenodo/Real:** F1 > 85% (real data variability)

### MAGIK + Gap-Filling (Tracking):

- **Track fragmentation:** 4-6× (acceptable)
- **Track length:** 20-40 detections
- **Completeness:** 70-85%

## Download Status

"""
    
    for dataset, status in results.items():
        if status == True:
            content += f"- ✅ **{dataset}**: Downloaded\n"
        elif status == 'skipped':
            content += f"- ⏭️ **{dataset}**: Skipped\n"
        else:
            content += f"- ❌ **{dataset}**: Failed (check logs)\n"
    
    content += """
## Dataset Sizes

| Dataset | Size | Frames | Particles | Download Time |
|---------|------|--------|-----------|---------------|
| SMLM Vesicles Low | ~100 MB | 300 | ~15/frame | 1-2 min |
| SMLM Tubules Low | ~100 MB | 300 | ~10/frame | 1-2 min |
| Tracking Receptors | ~50 MB | 500 | ~20/frame | 30 sec |
| Tracking Vesicles | ~60 MB | 500 | ~30/frame | 1 min |

## Citations

### SMLM Challenge:
> Sage D, Pham TA, Babcock H, et al. "Super-resolution fight club: Assessment of 
> 2D and 3D single-molecule localization microscopy software." Nature Methods 16, 
> 387–395 (2019).

### Particle Tracking Challenge:
> Chenouard N, Smal I, de Chaumont F, et al. "Objective comparison of particle 
> tracking methods." Nature Methods 11, 281–289 (2014).

### Zenodo Datasets:
Cite each dataset individually using its DOI.

## Notes

- All datasets are freely available for research
- Some may require manual download from websites
- Ground truth availability varies
- Real data (Zenodo) most similar to your PIEZO1 experiments

## Support

For issues with:
- **SMLM Challenge:** http://bigwww.epfl.ch/smlm/
- **Particle Tracking:** http://particletracking.github.io/
- **Zenodo:** https://zenodo.org/
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Created master README: {readme_path}")


if __name__ == '__main__':
    main()
