#!/usr/bin/env python3
"""
Download Zenodo Research Datasets

Downloads published particle tracking datasets from Zenodo.

Usage:
    python scripts/13_download_zenodo_datasets.py \
        --output data/public_datasets/zenodo \
        --datasets all
"""

import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm
import json


# Zenodo datasets - real research data
DATASETS = {
    'nino_vesicles': {
        'name': 'Nino et al. (2021) - Vesicle Tracking',
        'doi': '10.5281/zenodo.4568300',
        'zenodo_record': '4568300',
        'description': 'TIRF microscopy of vesicle exocytosis',
        'files': {
            # Example files - actual files may differ
            'example_movie': 'example_tirf_movie.tif',
        },
        'download_url': 'https://zenodo.org/record/4568300/files/',
        'paper': 'https://doi.org/10.1038/s41467-021-21935-7',
        'similar_to_piezo1': True,
        'notes': 'Visit Zenodo page to see available files and download manually if needed'
    },
    'manzo_spt': {
        'name': 'Manzo Lab - SPT Benchmark',
        'doi': '10.5281/zenodo.3707702',
        'zenodo_record': '3707702',
        'description': 'Single particle tracking benchmark datasets',
        'files': {
            'simulated_tracks': 'simulated_trajectories.csv',
        },
        'download_url': 'https://zenodo.org/record/3707702/files/',
        'paper': 'https://doi.org/10.1038/s41467-020-19160-7',
        'similar_to_piezo1': True,
        'notes': 'Benchmark for SPT analysis methods'
    },
    'jungmann_dnapaint': {
        'name': 'Jungmann Lab - DNA-PAINT',
        'doi': '10.5281/zenodo.3677575',
        'zenodo_record': '3677575',
        'description': 'DNA-PAINT super-resolution imaging',
        'files': {
            'example_data': 'dna_paint_example.tif',
        },
        'download_url': 'https://zenodo.org/record/3677575/files/',
        'paper': 'https://doi.org/10.1038/s41596-020-0292-x',
        'similar_to_piezo1': False,
        'notes': 'Different imaging modality but good for localization testing'
    }
}


def create_download_instructions(output_dir):
    """Create instructions for manual download."""
    output_dir = Path(output_dir)
    instructions_path = output_dir / 'DOWNLOAD_INSTRUCTIONS.md'
    
    content = """# Zenodo Dataset Download Instructions

## Why Manual Download?

Zenodo datasets often have multiple files and versions. It's best to:
1. Visit the Zenodo page
2. Browse available files
3. Download what you need

## Datasets

"""
    
    for key, dataset in DATASETS.items():
        zenodo_url = f"https://zenodo.org/record/{dataset['zenodo_record']}"
        
        content += f"""### {dataset['name']}

**DOI:** {dataset['doi']}  
**Zenodo Page:** {zenodo_url}  
**Paper:** {dataset['paper']}

**Description:** {dataset['description']}

**Similar to PIEZO1 data:** {'Yes' if dataset['similar_to_piezo1'] else 'No'}

**To download:**
1. Visit: {zenodo_url}
2. Click "Files" section
3. Download relevant .tif movies or localization data
4. Save to: `data/public_datasets/zenodo/{key}/`

**Notes:** {dataset['notes']}

---

"""
    
    content += """
## General Workflow

### 1. Download from Zenodo

```bash
# Create directory
mkdir -p data/public_datasets/zenodo/nino_vesicles

# Visit Zenodo page in browser
# Download files manually

# Or use wget/curl if you know the exact file URL:
wget -O data/public_datasets/zenodo/nino_vesicles/movie.tif \\
    https://zenodo.org/record/4568300/files/example.tif
```

### 2. Run DECODE+MAGIK Pipeline

```bash
python scripts/09_run_complete_pipeline.py \\
    --input data/public_datasets/zenodo/nino_vesicles/movie.tif \\
    --decode checkpoints/decode_optimized/best_model.pth \\
    --magik checkpoints/magik/best_model.pth \\
    --output results/zenodo_nino \\
    --use_gap_filling
```

### 3. Compare to Published Results

Many Zenodo datasets include:
- Processed localization data
- Tracked particles
- Analysis scripts

Compare your DECODE+MAGIK results to the published analysis!

## Recommended Datasets for PIEZO1 Comparison

### 1. Nino et al. Vesicles ⭐ (Most Similar)
- **Why:** TIRF microscopy, vesicle dynamics
- **Similarity:** Very high - punctate structures, blinking, diffusion
- **URL:** https://zenodo.org/record/4568300

### 2. Manzo SPT Benchmark ⭐ (Good for Validation)
- **Why:** Benchmark with ground truth
- **Similarity:** Good - single particle tracking
- **URL:** https://zenodo.org/record/3707702

### 3. SMALL-LABS Dataset
- **Why:** Cell surface receptors
- **Similarity:** Medium - similar motion patterns
- **URL:** Search Zenodo for "single molecule tracking"

## Finding More Datasets

Search Zenodo: https://zenodo.org/

Keywords:
- "single molecule tracking"
- "TIRF microscopy"
- "particle tracking"
- "super-resolution"
- "localization microscopy"

Filter by:
- File types: .tif, .csv
- License: Open (for research use)

## Citation

Always cite the original dataset when using:
```
@dataset{dataset_doi,
  author = {...},
  title = {...},
  year = {...},
  publisher = {Zenodo},
  doi = {dataset_doi}
}
```

## Notes

- Zenodo is the recommended repository for research data
- Datasets are peer-reviewed and persistent (DOI)
- Many include full analysis code
- Great for benchmarking your pipeline!
"""
    
    with open(instructions_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created: {instructions_path}")


def create_dataset_directories(output_dir):
    """Create directory structure for datasets."""
    output_dir = Path(output_dir)
    
    for key in DATASETS.keys():
        dataset_dir = output_dir / key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create info file
        info = DATASETS[key].copy()
        info_path = dataset_dir / 'dataset_info.json'
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        # Create placeholder README
        readme = f"""# {info['name']}

**DOI:** {info['doi']}  
**Zenodo:** https://zenodo.org/record/{info['zenodo_record']}

## Download Instructions

1. Visit: https://zenodo.org/record/{info['zenodo_record']}
2. Browse available files
3. Download .tif movies or data files
4. Save them to this directory

## Description

{info['description']}

**Paper:** {info['paper']}

## Usage

After downloading data files to this directory:

```bash
python scripts/09_run_complete_pipeline.py \\
    --input data/public_datasets/zenodo/{key}/your_movie.tif \\
    --decode checkpoints/decode_optimized/best_model.pth \\
    --magik checkpoints/magik/best_model.pth \\
    --output results/{key} \\
    --use_gap_filling
```

## Notes

{info['notes']}
"""
        
        readme_path = dataset_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme)


def main():
    parser = argparse.ArgumentParser(
        description='Setup Zenodo dataset directories and instructions'
    )
    parser.add_argument('--output', type=str,
                       default='data/public_datasets/zenodo',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ZENODO DATASET SETUP")
    print("="*70)
    print("\nCreating directory structure and download instructions...")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories and info files
    create_dataset_directories(output_dir)
    
    # Create download instructions
    create_download_instructions(output_dir)
    
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    
    print(f"\nCreated directories:")
    for key, dataset in DATASETS.items():
        print(f"  - {key}/ ({dataset['name']})")
    
    print(f"\n📖 Download instructions: {output_dir}/DOWNLOAD_INSTRUCTIONS.md")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Read: data/public_datasets/zenodo/DOWNLOAD_INSTRUCTIONS.md")
    print("2. Visit Zenodo pages and download data files")
    print("3. Save files to the appropriate directories")
    print("4. Run DECODE+MAGIK pipeline on the data")
    
    print("\n" + "="*70)
    print("RECOMMENDED DATASETS TO START:")
    print("="*70)
    for key, dataset in DATASETS.items():
        if dataset['similar_to_piezo1']:
            print(f"\n⭐ {dataset['name']}")
            print(f"   URL: https://zenodo.org/record/{dataset['zenodo_record']}")
            print(f"   Why: {dataset['description']}")


if __name__ == '__main__':
    main()
