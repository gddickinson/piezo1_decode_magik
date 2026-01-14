#!/usr/bin/env python3
"""
Download Particle Tracking Challenge Datasets

Downloads benchmark particle tracking datasets with ground truth.

Usage:
    python scripts/12_download_tracking_challenge.py \
        --output data/public_datasets/tracking_challenge \
        --datasets all
"""

import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm
import json


# Particle Tracking Challenge datasets
DATASETS = {
    'receptor_2d': {
        'name': 'Receptor Tracking 2D',
        'base_url': 'http://bioimageanalysis.org/track/datasets',
        'files': {
            'sequence': '01_Receptor.tif',
            'ground_truth': '01_GT.txt'
        },
        'description': 'Simulated membrane receptor diffusion',
        'frames': 500,
        'particles': '~20 per frame',
        'motion': 'Brownian diffusion',
        'has_gt': True
    },
    'vesicles_2d': {
        'name': 'Vesicle Tracking 2D',
        'base_url': 'http://bioimageanalysis.org/track/datasets',
        'files': {
            'sequence': '02_Vesicles.tif',
            'ground_truth': '02_GT.txt'
        },
        'description': 'Simulated vesicle trafficking',
        'frames': 500,
        'particles': '~30 per frame',
        'motion': 'Directed + diffusion',
        'has_gt': True
    },
    'microtubules_2d': {
        'name': 'Microtubule Tracking 2D',
        'base_url': 'http://bioimageanalysis.org/track/datasets',
        'files': {
            'sequence': '03_Microtubules.tif',
            'ground_truth': '03_GT.txt'
        },
        'description': 'Simulated microtubule plus-end tracking',
        'frames': 500,
        'particles': '~15 per frame',
        'motion': 'Directed growth',
        'has_gt': True
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                             desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                  reporthook=t.update_to)


def download_dataset(dataset_key, output_dir):
    """Download a dataset."""
    dataset = DATASETS[dataset_key]
    output_dir = Path(output_dir)
    dataset_dir = output_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Downloading: {dataset['name']}")
    print(f"{'='*70}")
    print(f"Description: {dataset['description']}")
    print(f"Frames: {dataset['frames']}")
    print(f"Particles: {dataset['particles']}")
    print(f"Motion type: {dataset['motion']}")
    print(f"{'='*70}\n")
    
    success = True
    
    # Download each file
    for file_type, filename in dataset['files'].items():
        url = f"{dataset['base_url']}/{filename}"
        output_path = dataset_dir / filename
        
        if output_path.exists():
            print(f"✓ Already downloaded: {filename}")
        else:
            print(f"Downloading {filename}...")
            try:
                download_file(url, output_path)
                print(f"✅ Downloaded: {filename}")
            except Exception as e:
                print(f"❌ Download failed: {e}")
                print(f"   URL: {url}")
                success = False
    
    # Save dataset info
    info = {
        'name': dataset['name'],
        'description': dataset['description'],
        'frames': dataset['frames'],
        'particles': dataset['particles'],
        'motion': dataset['motion'],
        'has_ground_truth': dataset['has_gt']
    }
    
    info_path = dataset_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    return success


def create_readme(output_dir):
    """Create README with dataset information."""
    output_dir = Path(output_dir)
    readme_path = output_dir / 'README.md'
    
    content = """# Particle Tracking Challenge Datasets

Downloaded from: http://particletracking.github.io/

## Datasets

"""
    
    for key, dataset in DATASETS.items():
        content += f"""### {dataset['name']} (`{key}`)

- **Description:** {dataset['description']}
- **Frames:** {dataset['frames']}
- **Particles:** {dataset['particles']}
- **Motion Type:** {dataset['motion']}
- **Ground Truth:** Available

"""
    
    content += """
## File Structure

Each dataset contains:
- `*.tif` - Microscopy movie (TIFF stack)
- `*_GT.txt` - Ground truth trajectories
- `dataset_info.json` - Dataset metadata

## Ground Truth Format

The `*_GT.txt` files contain tab-separated values:
```
particle_id    frame    x    y    z
```

Example:
```
1    0    125.3    87.2    0
1    1    126.1    88.1    0
2    0    45.7     123.4   0
...
```

## Usage with DECODE+MAGIK Pipeline

```bash
python scripts/09_run_complete_pipeline.py \\
    --input data/public_datasets/tracking_challenge/receptor_2d/01_Receptor.tif \\
    --decode checkpoints/decode_optimized/best_model.pth \\
    --magik checkpoints/magik/best_model.pth \\
    --output results/tracking_receptor \\
    --use_gap_filling
```

## Comparing to Ground Truth

After running the pipeline, compare your tracks to ground truth:

```python
import pandas as pd

# Load your tracks
your_tracks = pd.read_csv('results/tracking_receptor/tracks.csv')

# Load ground truth
gt = pd.read_csv('data/public_datasets/tracking_challenge/receptor_2d/01_GT.txt',
                 sep='\\t', names=['particle_id', 'frame', 'x', 'y', 'z'])

# Compare...
```

## Citation

If you use these datasets, please cite the Particle Tracking Challenge:

> Chenouard N, Smal I, de Chaumont F, et al. "Objective comparison of 
> particle tracking methods." Nature Methods 11, 281–289 (2014).
> https://doi.org/10.1038/nmeth.2808

## Notes

- These datasets simulate realistic particle tracking scenarios
- Ground truth includes particle IDs for track evaluation
- Useful for benchmarking tracking algorithms
- Similar dynamics to PIEZO1 puncta (diffusion + directed motion)
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Created README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download Particle Tracking Challenge datasets'
    )
    parser.add_argument('--output', type=str,
                       default='data/public_datasets/tracking_challenge',
                       help='Output directory')
    parser.add_argument('--datasets', type=str, default='all',
                       help='Datasets to download (comma-separated or "all")')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PARTICLE TRACKING CHALLENGE DATASET DOWNLOADER")
    print("="*70)
    
    # Parse dataset selection
    if args.datasets.lower() == 'all':
        selected = list(DATASETS.keys())
    else:
        selected = [d.strip() for d in args.datasets.split(',')]
        invalid = [d for d in selected if d not in DATASETS]
        if invalid:
            print(f"❌ Invalid datasets: {invalid}")
            print(f"Available: {list(DATASETS.keys())}")
            return
    
    print(f"\nSelected datasets: {selected}")
    print(f"Output directory: {args.output}\n")
    
    # Download
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = []
    failed = []
    
    for dataset_key in selected:
        if download_dataset(dataset_key, output_dir):
            success.append(dataset_key)
        else:
            failed.append(dataset_key)
    
    # Create README
    create_readme(output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✅ Successfully downloaded: {len(success)}")
    for d in success:
        print(f"   - {DATASETS[d]['name']}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for d in failed:
            print(f"   - {DATASETS[d]['name']}")
    
    print("\n" + "="*70)
    print("NOTES:")
    print("="*70)
    print("⚠️  Some datasets may not be publicly accessible")
    print("   If downloads fail, visit http://particletracking.github.io/")
    print("   to check availability or request access")


if __name__ == '__main__':
    main()
