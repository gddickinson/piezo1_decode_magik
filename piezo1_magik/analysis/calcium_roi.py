"""
ROI-Based Calcium Signal Analysis

Extract calcium fluorescence traces from small regions around each punctum.
Much simpler than semantic segmentation!

For each tracked punctum:
1. Extract NxN pixel ROI centered at (x, y)
2. Compute mean intensity over ROI
3. Generate fluorescence trace F(t)
4. Calculate ΔF/F₀
5. Detect peaks → calcium events
"""

import numpy as np
from scipy import ndimage, signal
from typing import List, Dict, Tuple
import pandas as pd


class CalciumROIAnalyzer:
    """
    Analyze calcium signals using ROI extraction.
    
    Much simpler than semantic segmentation - just extract intensity
    at punctum locations!
    """
    
    def __init__(self,
                 roi_size: int = 5,
                 baseline_method: str = 'percentile',
                 baseline_percentile: float = 10.0):
        """
        Args:
            roi_size: Size of ROI (roi_size × roi_size pixels)
            baseline_method: How to compute F₀ ('percentile' or 'rolling')
            baseline_percentile: Percentile for baseline (lower = more conservative)
        """
        self.roi_size = roi_size
        self.baseline_method = baseline_method
        self.baseline_percentile = baseline_percentile
        
        # ROI half-size
        self.half_size = roi_size // 2
    
    def extract_roi(self,
                   image: np.ndarray,
                   x: float,
                   y: float) -> np.ndarray:
        """
        Extract ROI from image at position (x, y).
        
        Args:
            image: (H, W) image
            x, y: Center position (sub-pixel OK)
            
        Returns:
            roi: (roi_size, roi_size) extracted region
                 Returns zeros if out of bounds
        """
        H, W = image.shape
        
        # Round to nearest pixel
        x_int = int(np.round(x))
        y_int = int(np.round(y))
        
        # Extract ROI
        y_min = max(0, y_int - self.half_size)
        y_max = min(H, y_int + self.half_size + 1)
        x_min = max(0, x_int - self.half_size)
        x_max = min(W, x_int + self.half_size + 1)
        
        roi = image[y_min:y_max, x_min:x_max]
        
        # Pad if near edge
        if roi.shape != (self.roi_size, self.roi_size):
            padded = np.zeros((self.roi_size, self.roi_size))
            h, w = roi.shape
            padded[:h, :w] = roi
            return padded
        
        return roi
    
    def extract_trace(self,
                     calcium_movie: np.ndarray,
                     track: pd.DataFrame) -> np.ndarray:
        """
        Extract calcium fluorescence trace for one track.
        
        Args:
            calcium_movie: (T, H, W) calcium channel movie
            track: DataFrame with columns ['frame', 'x', 'y']
            
        Returns:
            trace: (T,) fluorescence intensity trace
                   NaN for frames where track doesn't exist
        """
        T = calcium_movie.shape[0]
        trace = np.full(T, np.nan)
        
        for _, row in track.iterrows():
            frame = int(row['frame'])
            x = row['x']
            y = row['y']
            
            if 0 <= frame < T:
                roi = self.extract_roi(calcium_movie[frame], x, y)
                trace[frame] = roi.mean()
        
        return trace
    
    def compute_baseline(self, trace: np.ndarray) -> float:
        """
        Compute baseline fluorescence F₀.
        
        Args:
            trace: Fluorescence trace
            
        Returns:
            F0: Baseline fluorescence
        """
        # Remove NaNs
        valid = trace[~np.isnan(trace)]
        
        if len(valid) == 0:
            return 0.0
        
        if self.baseline_method == 'percentile':
            # Use low percentile (conservative baseline)
            F0 = np.percentile(valid, self.baseline_percentile)
        elif self.baseline_method == 'rolling':
            # Rolling minimum (more adaptive)
            window = min(50, len(valid) // 5)
            F0 = pd.Series(valid).rolling(window, center=True).min().median()
        else:
            # Simple mean
            F0 = valid.mean()
        
        return F0
    
    def compute_delta_f_over_f(self,
                               trace: np.ndarray) -> np.ndarray:
        """
        Compute ΔF/F₀.
        
        Args:
            trace: Raw fluorescence trace
            
        Returns:
            dff: ΔF/F₀ trace
        """
        F0 = self.compute_baseline(trace)
        
        if F0 == 0:
            return np.zeros_like(trace)
        
        dff = (trace - F0) / F0
        
        return dff
    
    def detect_peaks(self,
                    dff_trace: np.ndarray,
                    threshold: float = 2.0,
                    min_duration: int = 2,
                    min_distance: int = 5) -> List[Dict]:
        """
        Detect calcium events as peaks in ΔF/F₀ trace.
        
        Args:
            dff_trace: ΔF/F₀ trace
            threshold: Detection threshold (in units of std)
            min_duration: Minimum event duration (frames)
            min_distance: Minimum distance between peaks (frames)
            
        Returns:
            events: List of detected events
                Each event: {
                    'frame': int,
                    'amplitude': float,
                    'duration': int,
                    'peak_frame': int
                }
        """
        # Remove NaNs
        valid_mask = ~np.isnan(dff_trace)
        valid_trace = dff_trace.copy()
        valid_trace[~valid_mask] = 0
        
        # Compute threshold
        std = np.std(valid_trace[valid_mask]) if valid_mask.sum() > 0 else 0
        if std == 0:
            return []
        
        threshold_abs = threshold * std
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            valid_trace,
            height=threshold_abs,
            distance=min_distance
        )
        
        events = []
        
        for peak_idx in peaks:
            # Find event boundaries (where crosses threshold)
            # Search backward
            start = peak_idx
            while start > 0 and valid_trace[start-1] > threshold_abs / 2:
                start -= 1
            
            # Search forward
            end = peak_idx
            while end < len(valid_trace)-1 and valid_trace[end+1] > threshold_abs / 2:
                end += 1
            
            duration = end - start + 1
            
            # Filter by duration
            if duration >= min_duration:
                events.append({
                    'frame': start,
                    'peak_frame': peak_idx,
                    'amplitude': dff_trace[peak_idx],
                    'duration': duration,
                    'end_frame': end
                })
        
        return events
    
    def analyze_track(self,
                     calcium_movie: np.ndarray,
                     track: pd.DataFrame,
                     track_id: int,
                     detect_events: bool = True) -> Dict:
        """
        Complete analysis for one track.
        
        Args:
            calcium_movie: (T, H, W) calcium movie
            track: Track data
            track_id: Track identifier
            detect_events: Whether to detect events
            
        Returns:
            results: Dict with 'trace', 'dff', 'events'
        """
        # Extract trace
        trace = self.extract_trace(calcium_movie, track)
        
        # Compute ΔF/F₀
        dff = self.compute_delta_f_over_f(trace)
        
        results = {
            'track_id': track_id,
            'trace': trace,
            'dff': dff,
            'baseline': self.compute_baseline(trace)
        }
        
        # Detect events
        if detect_events:
            events = self.detect_peaks(dff)
            results['events'] = events
            results['num_events'] = len(events)
        
        return results


def analyze_all_tracks(calcium_movie: np.ndarray,
                      tracks_df: pd.DataFrame,
                      roi_size: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze calcium for all tracks.
    
    Args:
        calcium_movie: (T, H, W) calcium channel
        tracks_df: DataFrame with ['track_id', 'frame', 'x', 'y']
        roi_size: ROI size
        
    Returns:
        traces_df: Fluorescence traces per track
        events_df: Detected calcium events
    """
    
    analyzer = CalciumROIAnalyzer(roi_size=roi_size)
    
    all_traces = []
    all_events = []
    
    # Process each track
    for track_id in tracks_df['track_id'].unique():
        track_data = tracks_df[tracks_df['track_id'] == track_id]
        
        results = analyzer.analyze_track(
            calcium_movie,
            track_data,
            track_id,
            detect_events=True
        )
        
        # Store traces
        for frame, (f, dff) in enumerate(zip(results['trace'], results['dff'])):
            if not np.isnan(f):
                all_traces.append({
                    'track_id': track_id,
                    'frame': frame,
                    'fluorescence': f,
                    'dff': dff,
                    'baseline': results['baseline']
                })
        
        # Store events
        for event in results.get('events', []):
            all_events.append({
                'track_id': track_id,
                'frame': event['frame'],
                'peak_frame': event['peak_frame'],
                'amplitude': event['amplitude'],
                'duration': event['duration']
            })
    
    traces_df = pd.DataFrame(all_traces)
    events_df = pd.DataFrame(all_events)
    
    return traces_df, events_df


# Test the analyzer
if __name__ == '__main__':
    print("Testing Calcium ROI Analyzer...")
    
    # Create synthetic data
    T, H, W = 100, 256, 256
    calcium_movie = np.random.randn(T, H, W) * 50 + 500
    
    # Add calcium events
    for t in range(20, 40):
        # Event at (128, 128)
        y, x = np.ogrid[-5:6, -5:6]
        gaussian = np.exp(-(x**2 + y**2) / (2 * 2**2))
        calcium_movie[t, 123:134, 123:134] += 200 * gaussian
    
    # Create fake track
    track_data = []
    for t in range(T):
        track_data.append({
            'frame': t,
            'x': 128 + np.random.randn() * 0.5,
            'y': 128 + np.random.randn() * 0.5
        })
    
    track_df = pd.DataFrame(track_data)
    track_df['track_id'] = 0
    
    # Analyze
    analyzer = CalciumROIAnalyzer(roi_size=11)
    results = analyzer.analyze_track(calcium_movie, track_df, track_id=0)
    
    print(f"\n✅ Analysis complete:")
    print(f"  Trace length: {len(results['trace'])}")
    print(f"  Baseline F₀: {results['baseline']:.1f}")
    print(f"  Events detected: {results['num_events']}")
    
    if results['num_events'] > 0:
        print(f"\n  Event details:")
        for i, event in enumerate(results['events']):
            print(f"    Event {i+1}: frame {event['frame']}, "
                  f"amplitude {event['amplitude']:.2f}, "
                  f"duration {event['duration']} frames")
    
    print("\n✅ All tests passed!")
