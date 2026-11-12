"""Small synthetic dual-channel (PIEZO1 + calcium) demo movie for the web interface and the JS-vs-Python tests.

PIEZO1 channel: the repo's own generator (synthetic_generator.generate_synthetic_movie: Gaussian PSF, Poisson +
read noise, blinking), with ground-truth tracks. Calcium channel: built the way calcium_roi.py's self-test builds
one -- Gaussian noise around a constant level (default 500 +/- 50 ADU) plus calcium events, each a Gaussian spot
(amplitude 200 ADU, sigma 2 px) added for a run of consecutive frames -- except that here each event sits on a
tracked punctum and follows it. The result is rounded to uint16 like camera data.

This is synthetic demo data, not a measurement. Coordinates: pixel centres (x = column, y = row), as in the
generator and CalciumROIAnalyzer. Physical units: pixel size 130 nm (the PSF model default), no frame interval is
defined anywhere in the repo, so times are in frames.
"""
import numpy as np
import pandas as pd

from piezo1_magik.data.synthetic_generator import generate_synthetic_movie

DEMO_DEFAULTS = dict(seed=7, num_tracks=8, num_frames=40, image_size=(128, 128), calcium_level=500.0,
                     calcium_noise=50.0, event_amplitude=200.0, event_sigma_px=2.0, event_frames=8, num_events=4)
PIXEL_SIZE_NM = 130.0  # Gaussian2DPSF / Airy2DPSF default pixel_size_nm


def _positions_every_frame(track: pd.DataFrame, num_frames: int) -> np.ndarray:
    """(T, 2) x, y per frame: the visible detections, carried forward/backward over blink frames."""
    pos = np.full((num_frames, 2), np.nan)
    pos[track['frame'].to_numpy(int)] = track[['x', 'y']].to_numpy(float)
    s = pd.DataFrame(pos).ffill().bfill()
    return s.to_numpy()


def make_demo_dual_channel(**overrides):
    """Return dict(piezo1 (T,H,W) uint16, calcium (T,H,W) uint16, tracks DataFrame, events list, params)."""
    p = dict(DEMO_DEFAULTS, **overrides)
    T, (H, W) = p['num_frames'], p['image_size']
    np.random.seed(p['seed'])  # the generator uses NumPy's global RNG
    piezo1, tracks = generate_synthetic_movie(num_tracks=p['num_tracks'], num_frames=T, image_size=(H, W),
                                              with_blinking=True, output_dir=None)
    rng = np.random.default_rng(p['seed'])
    calcium = rng.normal(p['calcium_level'], p['calcium_noise'], size=(T, H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    ids = sorted(tracks['track_id'].unique())
    chosen = rng.choice(ids, size=min(p['num_events'], len(ids)), replace=False)
    events = []
    for tid in sorted(int(t) for t in chosen):
        start = int(rng.integers(3, T - p['event_frames'] - 3))
        pos = _positions_every_frame(tracks[tracks['track_id'] == tid], T)
        for t in range(start, start + p['event_frames']):
            x, y = pos[t]
            calcium[t] += p['event_amplitude'] * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * p['event_sigma_px'] ** 2))
        events.append({'track_id': tid, 'start_frame': start, 'end_frame': start + p['event_frames'] - 1})
    calcium = np.clip(np.round(calcium), 0, 65535).astype(np.uint16)
    return {'piezo1': piezo1, 'calcium': calcium, 'tracks': tracks.reset_index(drop=True), 'events': events,
            'params': p}
