"""Seeded simulation behind the default noise_window of CalciumROIAnalyzer.detect_peaks (configs/pipeline.yaml).

Synthetic ROI traces F(t) (no movie): baseline 500 ADU with Gaussian noise of 10 ADU (the noise of a 5 x 5 ROI mean
when pixels have sigma 50 ADU, as in the web demo), calcium events with an instant rise and exponential decay, and,
in the "drift" scenario, a slow baseline drift (linear ramp plus a slow sinusoid, bleaching/focus-like). Each trace
goes through the repo's own compute_delta_f_over_f and detect_peaks (defaults: percentile F0, threshold 2, min_duration
2, min_distance 5) for each candidate noise_window. Scoring per trace: a placed event is found when some detected peak
lies inside it (onset .. onset + 4 decay times); a detection whose peak lies in no placed event is a false positive;
F1 = 2 TP / (2 TP + FN + FP), pooled over all traces.

Run: python scripts/simulate_noise_window.py [--traces 200] [--frames 300] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piezo1_magik.analysis.calcium_roi import CalciumROIAnalyzer  # noqa: E402

WINDOWS = [0, 11, 21, 31, 41, 61, 81, 101, 151, 201]
F_BASE, NOISE = 500.0, 10.0  # ADU


def make_trace(rng, T, drift):
    """One F(t) trace and its placed events [(start, end)] (frames, inclusive)."""
    t = np.arange(T)
    f = np.full(T, F_BASE)
    if drift:
        ramp = rng.uniform(-1, 1) * rng.uniform(5, 20) * NOISE  # total change over the trace: 5-20 noise sigma
        period = rng.uniform(150, 600)
        f += ramp * t / T + rng.uniform(2, 8) * NOISE * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    events, cursor = [], int(rng.integers(5, 30))
    while cursor < T - 10:
        amp, tau = rng.uniform(3, 10) * NOISE, rng.uniform(1.5, 5)  # peak 3-10 noise sigma, decay 1.5-5 frames
        end = min(T - 1, cursor + int(np.ceil(4 * tau)))
        f[cursor:] += amp * np.exp(-(t[cursor:] - cursor) / tau)
        events.append((cursor, end))
        cursor = end + int(rng.integers(15, 80))
    return f + rng.normal(0, NOISE, T), events


def score(detections, events):
    found = sum(any(s <= d['peak_frame'] <= e for d in detections) for s, e in events)
    fp = sum(not any(s <= d['peak_frame'] <= e for s, e in events) for d in detections)
    return found, len(events) - found, fp


def run(n_traces, T, seed, windows=WINDOWS):
    """{scenario: {window: (TP, FN, FP, F1)}} for the 'flat' and 'drift' scenarios."""
    a = CalciumROIAnalyzer()
    out = {}
    for scenario, drift in (('flat', False), ('drift', True)):
        rng = np.random.default_rng(seed)
        traces = [make_trace(rng, T, drift) for _ in range(n_traces)]
        res = {}
        for w in windows:
            tp = fn = fp = 0
            for f, events in traces:
                k = score(a.detect_peaks(a.compute_delta_f_over_f(f), noise_window=w), events)
                tp, fn, fp = tp + k[0], fn + k[1], fp + k[2]
            res[w] = (tp, fn, fp, 2 * tp / (2 * tp + fn + fp))
        out[scenario] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--traces', type=int, default=200, help='traces per scenario')
    ap.add_argument('--frames', type=int, default=300, help='frames per trace')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    res = run(args.traces, args.frames, args.seed)
    print(f"{args.traces} traces x {args.frames} frames per scenario, seed {args.seed}")
    print(f"{'noise_window':>12} | {'flat: TP  FN  FP    F1':>24} | {'drift: TP  FN  FP    F1':>25}")
    for w in WINDOWS:
        cells = [f"{r[w][0]:5d} {r[w][1]:4d} {r[w][2]:4d} {r[w][3]:.3f}" for r in (res['flat'], res['drift'])]
        print(f"{w if w else '0 (whole)':>12} | {cells[0]:>24} | {cells[1]:>25}")


if __name__ == '__main__':
    main()
