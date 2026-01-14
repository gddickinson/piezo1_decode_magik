#!/usr/bin/env python3
"""
Download SMLM Challenge 2016 Datasets

Downloads benchmark single-molecule localization microscopy datasets.

Usage:
    python scripts/11_download_smlm_challenge.py \
        --output data/public_datasets/smlm_challenge \
        --datasets all
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd


# SMLM Challenge 2016 datasets
DATASETS = {
    'tubules_high': {
        'name': 'Bundled Tubules - High Density',
        'url': 'http://bigwww.epfl.ch/smlm/challenge2016/datasets/Data_SMLMS2016/Bundled_Tubes_High_Density.zip',
        'description': 'Simulated microtubule bundles, high density, challenging',
        'frames': 300,
        'particle_density': 'High (~50 particles/frame)',
        'has_gt': True
    },
    'tubules_low': {
        'name': 'Bundled Tubules - Low Density',
        'url': 'http://bigwww.epfl.ch/smlm/challenge2016/datasets/Data_SMLMS2016/Bundled_Tubes_Low_Density.zip',
        'description': 'Simulated microtubule bundles, low density, easier',
        'frames': 300,
        'particle_density': 'Low (~10 particles/frame)',
        'has_gt': True
    },
    'vesicles_high': {
        'name': 'Vesicles - High Density',
        'url': 'http://bigwww.epfl.ch/smlm/challenge2016/datasets/Data_SMLMS2016/Vesicles_High_Density.zip',
        'description': 'Simulated vesicles, high density',
        'frames': 300,
        'particle_density': 'High (~40 particles/frame)',
        'has_gt': True
    },
    'vesicles_low': {
        'name': 'Vesicles - Low Density',
        'url': 'http://bigwww.epfl.ch/smlm/challenge2016/datasets/Data_SMLMS2016/Vesicles_Low_Density.zip',
        'description': 'Simulated vesicles, low density, good for testing',
        'frames': 300,
        'particle_density': 'Low (~15 particles/frame)',
        'has_gt': True
    },
    'microtubules_exp': {
        'name': 'Microtubules - Experimental',
        'url': 'http://bigwww.epfl.ch/smlm/challenge2016/datasets/Data_SMLMS2016/MT0.45_DIL-HD.zip',
        'description': 'Real experimental microtubule data',
        'frames': 6000,
        'particle_density': 'Variable',
        'has_gt': False
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
    """Download and extract a dataset."""
    dataset = DATASETS[dataset_key]
    output_dir = Path(output_dir)
    dataset_dir = output_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Downloading: {dataset['name']}")
    print(f"{'='*70}")
    print(f"Description: {dataset['description']}")
    print(f"Frames: {dataset['frames']}")
    print(f"Density: {dataset['particle_density']}")
    print(f"Ground truth: {'Yes' if dataset['has_gt'] else 'No'}")
    print(f"{'='*70}\n")
    
    # Download zip file
    zip_path = dataset_dir / 'download.zip'
    
    if zip_path.exists():
        print(f"✓ Already downloaded: {zip_path}")
    else:
        print(f"Downloading from {dataset['url']}...")
        try:
            download_file(dataset['url'], zip_path)
            print(f"✅ Downloaded: {zip_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    # Extract
    print(f"Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print(f"✅ Extracted to: {dataset_dir}")
        
        # Clean up zip
        zip_path.unlink()
        print(f"✓ Cleaned up zip file")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False
    
    # List contents
    print(f"\nDataset contents:")
    for f in sorted(dataset_dir.rglob('*')):
        if f.is_file():
            size = f.stat().st_size / (1024*1024)  # MB
            print(f"  {f.name} ({size:.1f} MB)")
    
    return True


def create_dataset_info(output_dir):
    """Create README with dataset information."""
    output_dir = Path(output_dir)
    readme_path = output_dir / 'README.md'
    
    content = """# SMLM Challenge 2016 Datasets

Downloaded from: http://bigwww.epfl.ch/smlm/challenge2016/

## Datasets

"""
    
    for key, dataset in DATASETS.items():
        content += f"""### {dataset['name']} (`{key}`)

- **Description:** {dataset['description']}
- **Frames:** {dataset['frames']}
- **Particle Density:** {dataset['particle_density']}
- **Ground Truth:** {'Available' if dataset['has_gt'] else 'Not available'}

"""
    
    content += """
## File Structure

Each dataset typically contains:
- `sequence.tif` - Raw TIRF microscopy movie
- `groundtruth.csv` - Particle positions (for simulated data)
- `info.txt` - Dataset parameters

## Ground Truth Format

For datasets with ground truth, the CSV contains:
- `frame` - Frame number (0-indexed)
- `x`, `y` - Particle position in pixels
- `z` - Axial position (often 0 for TIRF)
- `id` - Particle/track ID

## Usage with DECODE+MAGIK Pipeline

Run the complete pipeline:

```bash
python scripts/09_run_complete_pipeline.py \\
    --input data/public_datasets/smlm_challenge/vesicles_low/sequence.tif \\
    --decode checkpoints/decode_optimized/best_model.pth \\
    --magik checkpoints/magik/best_model.pth \\
    --output results/smlm_vesicles \\
    --use_gap_filling
```

## Citation

If you use these datasets, please cite:

> Sage D, Pham TA, Babcock H, et al. "Super-resolution fight club: 
> Assessment of 2D and 3D single-molecule localization microscopy software." 
> Nature Methods 16, 387–395 (2019). 
> https://doi.org/10.1038/s41592-019-0364-4
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Created README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download SMLM Challenge 2016 datasets'
    )
    parser.add_argument('--output', type=str, 
                       default='data/public_datasets/smlm_challenge',
                       help='Output directory')
    parser.add_argument('--datasets', type=str, default='all',
                       help='Datasets to download (comma-separated or "all")')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SMLM CHALLENGE 2016 DATASET DOWNLOADER")
    print("="*70)
    
    # Parse dataset selection
    if args.datasets.lower() == 'all':
        selected = list(DATASETS.keys())
    else:
        selected = [d.strip() for d in args.datasets.split(',')]
        # Validate
        invalid = [d for d in selected if d not in DATASETS]
        if invalid:
            print(f"❌ Invalid datasets: {invalid}")
            print(f"Available: {list(DATASETS.keys())}")
            return
    
    print(f"\nSelected datasets: {selected}")
    print(f"Output directory: {args.output}\n")
    
    # Download each dataset
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
    create_dataset_info(output_dir)
    
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
    print("Next steps:")
    print("="*70)
    print(f"1. Check datasets in: {output_dir}")
    print(f"2. Read README: {output_dir}/README.md")
    print("3. Run pipeline on a dataset:")
    print(f"\n   python scripts/09_run_complete_pipeline.py \\")
    print(f"       --input {output_dir}/vesicles_low/sequence.tif \\")
    print(f"       --decode checkpoints/decode_optimized/best_model.pth \\")
    print(f"       --magik checkpoints/magik/best_model.pth \\")
    print(f"       --output results/test_public_data \\")
    print(f"       --use_gap_filling")


if __name__ == '__main__':
    main()
