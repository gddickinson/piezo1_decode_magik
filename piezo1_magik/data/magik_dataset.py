#!/usr/bin/env python3
"""
MAGIK Dataset for Graph-Based Tracking

Creates graph structures from detections for training MAGIK to link particles.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict
import tifffile


class MAGIKDataset(Dataset):
    """
    Dataset for training MAGIK tracking.
    
    Loads synthetic data, runs DECODE to get detections, builds graphs.
    """
    
    def __init__(self, data_dir: str, decode_model, device='cpu', 
                 max_temporal_gap=3, max_spatial_distance=50):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory with synthetic samples
            decode_model: Trained DECODE model for getting detections
            device: Device to run DECODE on
            max_temporal_gap: Maximum frame gap for linking (typically 1-3)
            max_spatial_distance: Maximum distance (pixels) for linking
        """
        self.data_dir = Path(data_dir)
        self.decode_model = decode_model
        self.device = device
        self.max_temporal_gap = max_temporal_gap
        self.max_spatial_distance = max_spatial_distance
        
        # Find all samples
        self.samples = sorted(list(self.data_dir.glob('sample_*')))
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {data_dir}")
        
        print(f"Found {len(self.samples)} samples for MAGIK training")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get one training sample.
        
        Returns:
            dict with:
                - node_features: (N, F) detection features
                - edge_index: (2, E) graph edges
                - edge_features: (E, F_e) edge features
                - edge_labels: (E,) binary labels (1=same track, 0=different)
                - node_tracks: (N,) ground truth track IDs
        """
        sample_dir = self.samples[idx]
        
        # Load movie
        movie_path = sample_dir / 'movie.tif'
        movie = tifffile.imread(movie_path)  # (T, H, W)
        T, H, W = movie.shape
        
        # Load ground truth tracks
        gt_path = sample_dir / 'ground_truth_tracks.csv'
        gt_df = pd.read_csv(gt_path)
        
        # Run DECODE on all frames to get detections
        detections = self._run_decode_on_movie(movie)
        
        # Match detections to ground truth to get track IDs
        detections_with_tracks = self._match_to_ground_truth(detections, gt_df)
        
        # Build graph
        graph_data = self._build_graph(detections_with_tracks)
        
        return graph_data
    
    def _run_decode_on_movie(self, movie: np.ndarray) -> List[Dict]:
        """
        Run DECODE on all frames to get detections.
        
        Args:
            movie: (T, H, W) movie array (uint16)
            
        Returns:
            List of detections, each dict with:
                - frame: int
                - x: float
                - y: float
                - prob: float
                - photons: float
        """
        T, H, W = movie.shape
        all_detections = []
        
        self.decode_model.eval()
        
        with torch.no_grad():
            # Process with 3-frame temporal windows
            for t in range(1, T - 1):
                # Get 3-frame window
                window = movie[t-1:t+2]  # (3, H, W)
                
                # Normalize
                window = window.astype(np.float32) / 65535.0
                
                # To tensor
                window_tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)  # (1, 3, H, W)
                
                # Run DECODE
                outputs = self.decode_model(window_tensor)
                
                # Get predictions
                prob = outputs['prob'][0, 0].cpu().numpy()  # (H, W)
                offset = outputs['offset'][0].cpu().numpy()  # (2, H, W)
                photons = outputs['photons'][0, 0].cpu().numpy()  # (H, W)
                
                # Find detections (threshold at 0.3 for training - lower to get more examples)
                threshold = 0.3
                det_mask = prob > threshold
                det_y, det_x = np.where(det_mask)
                
                # Extract detections for this frame
                for y, x in zip(det_y, det_x):
                    all_detections.append({
                        'frame': t,
                        'x': x + offset[0, y, x],
                        'y': y + offset[1, y, x],
                        'prob': prob[y, x],
                        'photons': photons[y, x]
                    })
        
        return all_detections
    
    def _match_to_ground_truth(self, detections: List[Dict], gt_df: pd.DataFrame) -> List[Dict]:
        """
        Match detections to ground truth tracks.
        
        Args:
            detections: List of detection dicts
            gt_df: Ground truth DataFrame with columns: frame, track_id, x, y, photons
            
        Returns:
            List of detections with added 'track_id' field
        """
        detections_with_tracks = []
        
        for det in detections:
            frame = det['frame']
            det_x = det['x']
            det_y = det['y']
            
            # Find ground truth detections in this frame
            gt_frame = gt_df[gt_df['frame'] == frame]
            
            if len(gt_frame) == 0:
                continue  # No GT in this frame
            
            # Find nearest GT detection
            distances = np.sqrt((gt_frame['x'] - det_x)**2 + (gt_frame['y'] - det_y)**2)
            min_dist = distances.min()
            
            # If within 3 pixels, assign track ID
            if min_dist < 3.0:
                nearest_idx = distances.idxmin()
                track_id = gt_frame.loc[nearest_idx, 'track_id']
                
                det_with_track = det.copy()
                det_with_track['track_id'] = track_id
                detections_with_tracks.append(det_with_track)
        
        return detections_with_tracks
    
    def _build_graph(self, detections: List[Dict]) -> Dict:
        """
        Build temporal graph from detections.
        
        Args:
            detections: List of detections with track_id
            
        Returns:
            dict with graph data
        """
        if len(detections) < 2:
            # Return empty graph
            return {
                'node_features': torch.zeros((0, 5), dtype=torch.float32),
                'edge_index': torch.zeros((2, 0), dtype=torch.long),
                'edge_features': torch.zeros((0, 4), dtype=torch.float32),
                'edge_labels': torch.zeros((0,), dtype=torch.float32),
                'node_tracks': torch.zeros((0,), dtype=torch.long)
            }
        
        # Create node features (N, 5): [frame, x, y, prob, photons]
        N = len(detections)
        node_features = np.zeros((N, 5), dtype=np.float32)
        node_tracks = np.zeros(N, dtype=np.int64)
        
        for i, det in enumerate(detections):
            node_features[i] = [
                det['frame'],
                det['x'],
                det['y'],
                det['prob'],
                det['photons']
            ]
            node_tracks[i] = det['track_id']
        
        # Build edges - connect detections across nearby frames
        edge_list = []
        edge_features_list = []
        edge_labels_list = []
        
        # Group detections by frame
        frame_to_nodes = {}
        for i, det in enumerate(detections):
            frame = det['frame']
            if frame not in frame_to_nodes:
                frame_to_nodes[frame] = []
            frame_to_nodes[frame].append(i)
        
        frames = sorted(frame_to_nodes.keys())
        
        # Connect each frame to next few frames
        for i, frame in enumerate(frames):
            for j in range(i + 1, min(i + 1 + self.max_temporal_gap, len(frames))):
                next_frame = frames[j]
                temporal_gap = next_frame - frame
                
                if temporal_gap > self.max_temporal_gap:
                    break
                
                # Connect all pairs within distance threshold
                for src_node in frame_to_nodes[frame]:
                    src_x = node_features[src_node, 1]
                    src_y = node_features[src_node, 2]
                    
                    for dst_node in frame_to_nodes[next_frame]:
                        dst_x = node_features[dst_node, 1]
                        dst_y = node_features[dst_node, 2]
                        
                        # Compute distance
                        dx = dst_x - src_x
                        dy = dst_y - src_y
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        # Only connect if within distance threshold
                        if distance <= self.max_spatial_distance:
                            # Add edge
                            edge_list.append([src_node, dst_node])
                            
                            # Edge features: [dx, dy, distance, temporal_gap]
                            edge_features_list.append([dx, dy, distance, temporal_gap])
                            
                            # Label: 1 if same track, 0 otherwise
                            same_track = (node_tracks[src_node] == node_tracks[dst_node])
                            edge_labels_list.append(1.0 if same_track else 0.0)
        
        # Convert to tensors
        if len(edge_list) == 0:
            # No edges - return empty graph
            return {
                'node_features': torch.from_numpy(node_features),
                'edge_index': torch.zeros((2, 0), dtype=torch.long),
                'edge_features': torch.zeros((0, 4), dtype=torch.float32),
                'edge_labels': torch.zeros((0,), dtype=torch.float32),
                'node_tracks': torch.from_numpy(node_tracks)
            }
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()  # (2, E)
        edge_features = torch.tensor(edge_features_list, dtype=torch.float32)
        edge_labels = torch.tensor(edge_labels_list, dtype=torch.float32)
        
        return {
            'node_features': torch.from_numpy(node_features),
            'edge_index': edge_index,
            'edge_features': edge_features,
            'edge_labels': edge_labels,
            'node_tracks': torch.from_numpy(node_tracks)
        }


def collate_graphs(batch: List[Dict]) -> Dict:
    """
    Collate multiple graphs into a single batch.
    
    Uses disjoint union of graphs.
    """
    # Combine all graphs
    node_features_list = []
    edge_index_list = []
    edge_features_list = []
    edge_labels_list = []
    node_tracks_list = []
    
    node_offset = 0
    
    for sample in batch:
        node_features_list.append(sample['node_features'])
        node_tracks_list.append(sample['node_tracks'])
        edge_features_list.append(sample['edge_features'])
        edge_labels_list.append(sample['edge_labels'])
        
        # Offset edge indices
        edge_index = sample['edge_index'] + node_offset
        edge_index_list.append(edge_index)
        
        node_offset += sample['node_features'].shape[0]
    
    # Concatenate
    return {
        'node_features': torch.cat(node_features_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'edge_features': torch.cat(edge_features_list, dim=0),
        'edge_labels': torch.cat(edge_labels_list, dim=0),
        'node_tracks': torch.cat(node_tracks_list, dim=0)
    }
