"""
Synthetic Data Generator with Ground Truth Tracks

Generates realistic PIEZO1-HaloTag movies with:
- Multiple puncta trajectories
- Realistic PSF
- Blinking dynamics
- Ground truth tracks for MAGIK training
"""

import numpy as np
import tifffile
import pandas as pd
from pathlib import Path
import json
from typing import List, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from piezo1_magik.utils.psf_models import Gaussian2DPSF, add_noise


class TrackGenerator:
    """Generate realistic particle trajectories."""
    
    def __init__(self, image_size=(512, 512), num_frames=100):
        self.H, self.W = image_size
        self.T = num_frames
    
    def generate_tracks(self,
                       num_tracks=10,
                       diffusion_coeff=0.5,
                       directed_motion_prob=0.3,
                       directed_speed=2.0) -> List[np.ndarray]:
        """
        Generate particle tracks.
        
        Args:
            num_tracks: Number of trajectories
            diffusion_coeff: Diffusion coefficient (pixels/frame)
            directed_motion_prob: Probability of directed motion
            directed_speed: Speed for directed motion (pixels/frame)
            
        Returns:
            tracks: List of (T, 2) arrays with (x, y) positions
        """
        tracks = []
        
        for _ in range(num_tracks):
            # Random starting position (avoid edges)
            x0 = np.random.uniform(50, self.W - 50)
            y0 = np.random.uniform(50, self.H - 50)
            
            # Initialize trajectory
            positions = np.zeros((self.T, 2))
            positions[0] = [x0, y0]
            
            # Determine motion type
            is_directed = np.random.rand() < directed_motion_prob
            
            if is_directed:
                # Directed motion with diffusion
                direction = np.random.rand() * 2 * np.pi
                velocity = directed_speed * np.array([np.cos(direction), np.sin(direction)])
                
                for t in range(1, self.T):
                    # Directed + diffusion
                    diffusion = np.random.randn(2) * diffusion_coeff
                    positions[t] = positions[t-1] + velocity + diffusion
            else:
                # Pure diffusion
                for t in range(1, self.T):
                    diffusion = np.random.randn(2) * diffusion_coeff
                    positions[t] = positions[t-1] + diffusion
            
            # Clip to image bounds
            positions[:, 0] = np.clip(positions[:, 0], 0, self.W - 1)
            positions[:, 1] = np.clip(positions[:, 1], 0, self.H - 1)
            
            tracks.append(positions)
        
        return tracks
    
    def add_blinking(self, tracks: List[np.ndarray],
                    blink_prob=0.1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Add blinking to tracks.
        
        Args:
            tracks: Original tracks
            blink_prob: Probability of blinking per frame
            
        Returns:
            blinking_tracks: Tracks with blinking
            visibility: Visibility mask per track
        """
        blinking_tracks = []
        visibility_masks = []
        
        for track in tracks:
            # Generate visibility mask
            visible = np.random.rand(len(track)) > blink_prob
            
            # Keep track but mark invisible frames
            blinking_tracks.append(track)
            visibility_masks.append(visible)
        
        return blinking_tracks, visibility_masks


def generate_synthetic_movie(num_tracks=10,
                            num_frames=100,
                            image_size=(512, 512),
                            photons_mean=2000,  # Increased from 1000
                            photons_std=400,     # Increased from 200
                            baseline=100,        # Camera baseline
                            read_noise=10,       # Read noise
                            with_blinking=True,
                            output_dir=None):
    """
    Generate complete synthetic movie with tracks.
    
    Args:
        num_tracks: Number of particle tracks
        num_frames: Number of frames
        image_size: Image size (H, W)
        photons_mean: Mean photon count
        photons_std: Photon count variation
        with_blinking: Add blinking dynamics
        output_dir: Output directory (if None, return data)
        
    Returns:
        movie: (T, H, W) movie
        tracks_df: DataFrame with ground truth tracks
    """
    
    H, W = image_size
    T = num_frames
    
    # Generate tracks
    track_gen = TrackGenerator(image_size=image_size, num_frames=num_frames)
    tracks = track_gen.generate_tracks(num_tracks=num_tracks)
    
    # Add blinking
    if with_blinking:
        tracks, visibility = track_gen.add_blinking(tracks, blink_prob=0.1)
    else:
        visibility = [np.ones(len(track), dtype=bool) for track in tracks]
    
    # Create PSF model
    psf_model = Gaussian2DPSF()
    
    # Generate movie
    movie = np.zeros((T, H, W), dtype=np.float32)
    
    all_detections = []
    
    for track_id, (track, vis) in enumerate(zip(tracks, visibility)):
        for t in range(T):
            if vis[t]:
                x, y = track[t]
                
                # Random photon count
                photons = max(100, np.random.normal(photons_mean, photons_std))
                
                # Generate PSF
                psf = psf_model.generate(x, y, photons=photons, image_size=image_size)
                
                # Add to movie
                movie[t] += psf
                
                # Record detection
                all_detections.append({
                    'track_id': track_id,
                    'frame': t,
                    'x': x,
                    'y': y,
                    'photons': photons,
                    'sigma_x': psf_model.sigma_px,
                    'sigma_y': psf_model.sigma_px
                })
    
    # Add noise
    movie_noisy = np.zeros((T, H, W), dtype=np.uint16)
    for t in range(T):
        movie_noisy[t] = add_noise(movie[t], baseline=baseline, read_noise=read_noise)
    
    # Create DataFrame
    tracks_df = pd.DataFrame(all_detections)
    
    # Save if output_dir specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save movie
        tifffile.imwrite(output_dir / 'movie.tif', movie_noisy)
        
        # Save tracks
        tracks_df.to_csv(output_dir / 'ground_truth_tracks.csv', index=False)
        
        # Save metadata
        metadata = {
            'num_tracks': num_tracks,
            'num_frames': num_frames,
            'image_size': list(image_size),
            'photons_mean': photons_mean,
            'photons_std': photons_std,
            'baseline': baseline,
            'read_noise': read_noise,
            'with_blinking': with_blinking
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved to {output_dir}")
        print(f"   Movie: {movie_noisy.shape}")
        print(f"   Detections: {len(tracks_df)}")
        print(f"   Tracks: {num_tracks}")
    
    return movie_noisy, tracks_df


# Test
if __name__ == '__main__':
    print("Testing synthetic data generator...")
    
    movie, tracks = generate_synthetic_movie(
        num_tracks=5,
        num_frames=50,
        image_size=(256, 256),
        with_blinking=True,
        output_dir='/tmp/test_synthetic'
    )
    
    print(f"\n✅ Generated synthetic data:")
    print(f"   Movie shape: {movie.shape}")
    print(f"   Total detections: {len(tracks)}")
    print(f"   Unique tracks: {tracks['track_id'].nunique()}")
    print(f"   Frames per track: {tracks.groupby('track_id').size().describe()}")
