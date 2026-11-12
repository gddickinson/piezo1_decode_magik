"""Write tests/fixtures/engine_cases.json: inputs and the Python code's outputs for the JS engine tests.

piezo1_magik/web/static/{roi,track,psf}_engine.test.mjs run the JS ports on the same inputs and check they reproduce
these numbers; tests/test_web.py checks the fixture still matches what the Python code computes now.

Run: python scripts/make_engine_fixture.py
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from piezo1_magik.analysis.calcium_roi import CalciumROIAnalyzer  # noqa: E402
from piezo1_magik.tracking.linking import greedy_linking  # noqa: E402
from piezo1_magik.utils.psf_models import Airy2DPSF, Gaussian2DPSF  # noqa: E402
from piezo1_magik.web.demo import clean, demo, roi_on, run_tracks  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_cases.json"
SHAPE = (50, 24, 24)
ROI_PARAMS = [
    {},
    dict(roi_size=1), dict(roi_size=3, threshold=1.0, min_duration=1, min_distance=1),
    dict(roi_size=7, baseline_method='rolling'), dict(baseline_method='mean', threshold=1.5),
    dict(noise_window=0), dict(noise_window=1), dict(noise_window=4, threshold=1.0, min_duration=1, min_distance=1),
    dict(roi_size=3, noise_window=11, threshold=1.5), dict(noise_window=21, baseline_method='rolling'),
    dict(baseline_percentile=50.0, min_duration=3, min_distance=8), dict(roi_size=9, baseline_method='rolling', threshold=0.5),
]
TRACK_PARAMS = [{}, dict(max_temporal_gap=1, max_spatial_distance=3.0, max_gap=3, max_distance=2.0),
                dict(max_temporal_gap=3, max_spatial_distance=8.0, max_gap=20, max_distance=60.0),
                dict(max_temporal_gap=1, max_spatial_distance=40.0, max_gap=2, max_distance=5.0),
                dict(max_temporal_gap=1, max_spatial_distance=0.3, max_gap=10, max_distance=10.0)]  # wrong merges
PSF_CASES = [dict(opts={}, x=7.0, y=7.0, photons=1000.0, H=15, W=15),
             dict(opts=dict(wavelength_nm=520, NA=1.2, pixel_size_nm=108), x=5.3, y=6.8, photons=2000.0, H=13, W=11),
             dict(opts=dict(wavelength_nm=700, NA=0.9, pixel_size_nm=65), x=10.5, y=4.25, photons=500.0, H=12, W=21)]


def roi_inputs():
    """Calcium-like movie (500 +/- 50 ADU with two events) and tracks covering the edge cases of extract_roi."""
    rng = np.random.default_rng(11)
    T, H, W = SHAPE
    movie = rng.normal(500, 50, SHAPE)
    yy, xx = np.mgrid[0:H, 0:W]
    for t in range(20, 27):
        movie[t] += 300 * np.exp(-((xx - 12) ** 2 + (yy - 12) ** 2) / 8)
    for t in range(5, 9):
        movie[t] += 250 * np.exp(-((xx - 1) ** 2 + (yy - 22) ** 2) / 8)
    movie = np.clip(np.round(movie), 0, 65535).astype(np.uint16)
    rows = []
    for f in range(T):
        rows.append((0, f, 12 + 0.3 * np.sin(f), 12 + 0.3 * np.cos(f)))  # centre, sub-pixel jitter
        if f % 4:
            rows.append((1, f, 0.6, 22.4))  # at the image edge, with gaps
    rows += [(2, f, 2.5 + f % 2, 3.5) for f in range(10, 30)]  # exact .5 positions: round half to even
    rows += [(3, f, -10.0, 5.0) for f in range(3)]  # outside the image, too short for a rolling window
    rows += [(4, f, 23.49, 0.51) for f in range(0, 50, 2)]  # corner, every other frame
    rows += [(5, 60, 5.0, 5.0)]  # frame beyond the movie: ignored
    return movie, pd.DataFrame(rows, columns=['track_id', 'frame', 'x', 'y'])


def peak_cases():
    rng = np.random.default_rng(3)
    cases = []
    for k in range(25):
        x = np.round(rng.normal(0, 1, int(rng.integers(5, 80))), 1 if k % 2 else 3)  # rounding makes plateaus/ties
        h, d = float(rng.uniform(-0.5, 1.5)), int(rng.integers(1, 8))
        cases.append({'x': x.tolist(), 'height': h, 'distance': d, 'peaks': find_peaks(x, height=h, distance=d)[0].tolist()})
    return cases


def noise_cases():
    """noise_levels/detect_peaks with a rolling window: a drifting dF/F0 trace with events and NaN gaps, a short trace,
    a quantised trace (local MAD 0 -> whole-trace values)."""
    rng = np.random.default_rng(9)
    t = np.arange(150)
    drift = 0.002 * t + 0.08 * np.sin(2 * np.pi * t / 90) + rng.normal(0, 0.02, 150)
    for s in (20, 70, 120):
        drift[s:] += 0.15 * np.exp(-(t[s:] - s) / 3)
    drift[[5, 6, 7, 60, 100, 101]] = np.nan
    short = rng.normal(0, 0.05, 9)
    short[4] = 0.4
    quant = np.round(rng.normal(0, 0.6, 60)) / 10
    quant[30:33] += 0.5
    a = CalciumROIAnalyzer()
    cases = []
    for dff in (drift, short, quant):
        for w in (0, 1, 3, 10, 31, 61):
            level, sigma = a.noise_levels(dff, w)
            cases.append({'dff': dff.tolist(), 'window': w, 'level': level.tolist(), 'sigma': sigma.tolist(),
                          'events': a.detect_peaks(dff, threshold=2.0, min_duration=2, min_distance=3, noise_window=w)})
    return cases


def link_cases():
    rng = np.random.default_rng(5)
    cases = []
    for _ in range(10):
        n, e = int(rng.integers(4, 30)), int(rng.integers(3, 60))
        src = rng.integers(0, n - 1, e)
        dst = np.minimum(src + rng.integers(1, 4, e), n - 1)
        probs = np.round(rng.uniform(0, 1, e), 2)  # rounding gives ties
        n2t = greedy_linking(probs, np.vstack([src, dst]), 0.4)
        cases.append({'src': src.tolist(), 'dst': dst.tolist(), 'probs': probs.tolist(),
                      'result': {str(k): v for k, v in n2t.items()}})
    return cases


def build() -> dict:
    movie, tracks = roi_inputs()
    d = demo()['tracks']
    sub = d[d['frame'] < 20].reset_index(drop=True)
    psf = []
    for c in PSF_CASES:
        g, a = Gaussian2DPSF(**c['opts']), Airy2DPSF(**c['opts'])
        psf.append({**c, 'sigma_px': g.sigma_px, 'gauss': g.generate(c['x'], c['y'], c['photons'], (c['H'], c['W'])),
                    'airy': a.generate(c['x'], c['y'], c['photons'], (c['H'], c['W']))})
    return clean({
        'roi': {'shape': list(SHAPE), 'movie_b64': base64.b64encode(movie.astype('<u2').tobytes()).decode(),
                'tracks': {c: tracks[c].tolist() for c in tracks.columns},
                'runs': [roi_on(movie, tracks, p) for p in ROI_PARAMS]},
        'peaks': peak_cases(),
        'noise': noise_cases(),
        'tracks': {'input': {c: sub[c].tolist() for c in ('track_id', 'frame', 'x', 'y')},
                   'runs': [run_tracks(p, tracks=sub) for p in TRACK_PARAMS]},
        'links': link_cases(),
        'psf': psf,
    })


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
