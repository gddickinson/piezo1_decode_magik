"""
PyTorch Dataset for DECODE Training

Loads synthetic data with 3-frame windows for DECODE training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import tifffile
import pandas as pd
from typing import Tuple


class DECODEDataset(Dataset):
    """
    Dataset for training DECODE.
    
    Loads synthetic movies and creates 3-frame windows with ground truth.
    """
    
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Directory with synthetic samples
            transform: Optional data augmentation
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all samples
        self.samples = sorted(list(self.data_dir.glob('sample_*')))
        
        print(f"Found {len(self.samples)} samples in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Load one sample.
        
        Returns:
            dict with:
                'images': (3, H, W) three-frame window (center frame)
                'has_puncta': (1, H, W) detection mask
                'offset': (2, H, W) sub-pixel offsets
                'photons': (1, H, W) photon counts
        """
        sample_dir = self.samples[idx]
        
        # Load movie
        movie = tifffile.imread(sample_dir / 'movie.tif')  # (T, H, W)
        
        # Load ground truth
        gt_df = pd.read_csv(sample_dir / 'ground_truth_tracks.csv')
        
        # Choose random center frame (not first or last)
        T = movie.shape[0]
        center_frame = np.random.randint(1, T - 1)
        
        # Extract 3-frame window
        window = movie[center_frame-1:center_frame+2]  # (3, H, W)
        
        # Normalize to [0, 1]
        window = window.astype(np.float32) / 65535.0
        
        # Create ground truth for center frame
        H, W = window.shape[1:]
        has_puncta = np.zeros((H, W), dtype=np.float32)
        offset_x = np.zeros((H, W), dtype=np.float32)
        offset_y = np.zeros((H, W), dtype=np.float32)
        photons = np.zeros((H, W), dtype=np.float32)
        
        # Get puncta in center frame
        frame_puncta = gt_df[gt_df['frame'] == center_frame]
        
        for _, row in frame_puncta.iterrows():
            x = row['x']
            y = row['y']
            p = row['photons']
            
            # Pixel coordinates
            px = int(np.floor(x))
            py = int(np.floor(y))
            
            if 0 <= px < W and 0 <= py < H:
                has_puncta[py, px] = 1.0
                offset_x[py, px] = x - px  # Sub-pixel offset
                offset_y[py, px] = y - py
                photons[py, px] = p
        
        # Convert to tensors
        images = torch.from_numpy(window)  # (3, H, W)
        has_puncta = torch.from_numpy(has_puncta).unsqueeze(0)  # (1, H, W)
        offset = torch.stack([
            torch.from_numpy(offset_x),
            torch.from_numpy(offset_y)
        ], dim=0)  # (2, H, W)
        photons = torch.from_numpy(photons).unsqueeze(0)  # (1, H, W)
        
        # Apply transforms if any
        if self.transform:
            # TODO: Implement data augmentation
            pass
        
        return {
            'images': images,
            'has_puncta': has_puncta,
            'offset': offset,
            'photons': photons
        }


def create_dataloaders(data_dir: str,
                      batch_size: int = 4,
                      train_split: float = 0.9,
                      num_workers: int = 4,
                      device: str = 'cpu') -> Tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory with synthetic samples
        batch_size: Batch size
        train_split: Fraction for training
        num_workers: Number of data loading workers
        device: Device being used (for pin_memory setting)
        
    Returns:
        train_loader, val_loader
    """
    
    # Create full dataset
    full_dataset = DECODEDataset(data_dir)
    
    # Split train/val
    n_train = int(len(full_dataset) * train_split)
    n_val = len(full_dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Disable pin_memory for MPS (not supported)
    use_pin_memory = 'cuda' in device
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader


# Test the dataset
if __name__ == '__main__':
    import sys
    
    print("Testing DECODE Dataset...")
    
    # Test with synthetic data if it exists
    test_dir = 'data/synthetic'
    
    if Path(test_dir).exists():
        dataset = DECODEDataset(test_dir)
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Load one sample
        sample = dataset[0]
        
        print(f"\n✅ Sample loaded:")
        print(f"   Images shape: {sample['images'].shape}")
        print(f"   Has puncta: {sample['has_puncta'].sum().item()} pixels")
        print(f"   Offset shape: {sample['offset'].shape}")
        print(f"   Photons range: {sample['photons'].min():.1f} - {sample['photons'].max():.1f}")
        
        # Test dataloader
        train_loader, val_loader = create_dataloaders(test_dir, batch_size=2)
        
        print(f"\n✅ Dataloaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        # Load one batch
        batch = next(iter(train_loader))
        print(f"\n✅ Batch loaded:")
        print(f"   Images: {batch['images'].shape}")
        print(f"   Has puncta: {batch['has_puncta'].shape}")
    else:
        print(f"⚠️  Test directory not found: {test_dir}")
        print("   Generate synthetic data first with:")
        print("   python scripts/01_generate_synthetic_data.py --output data/synthetic")
