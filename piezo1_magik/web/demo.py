"""Synthetic demo dataset (piezo1_magik.data.demo_dual_channel) as JSON, and the Python side of the two in-browser
analyses: calcium ROI analysis (CalciumROIAnalyzer) and track post-processing (piezo1_magik.tracking).

The same computations run in the browser on the static site (static/roi_engine.js, static/track_engine.js);
scripts/make_engine_fixture.py and the node tests check that both give the same numbers.
"""
import base64
from functools import lru_cache

import numpy as np

from piezo1_magik.analysis.calcium_roi import CalciumROIAnalyzer
from piezo1_magik.data.demo_dual_channel import PIXEL_SIZE_NM, make_demo_dual_channel
from piezo1_magik.tracking.linking import (build_candidate_graph, build_track_fragments, fragments_to_node_mapping,
                                          greedy_linking, merge_track_fragments)
from piezo1_magik.tracking.track_stats import analyze_fragmentation, analyze_track_lengths, length_categories

ROI_DEFAULTS = dict(roi_size=5, baseline_method='percentile', baseline_percentile=10.0, threshold=2.0,
                    min_duration=2, min_distance=5)  # CalciumROIAnalyzer / detect_peaks defaults
TRACK_DEFAULTS = dict(max_temporal_gap=2, max_spatial_distance=30.0, max_gap=10, max_distance=30.0)
# graph: configs/magik_training.yaml; gap filling: scripts/08_fill_track_gaps.py defaults


def clean(x):
    """JSON-safe: NaN/inf -> None, NumPy scalars -> Python."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


@lru_cache(maxsize=1)
def demo():
    return make_demo_dual_channel()


def demo_json():
    d = demo()
    p1 = d['piezo1'].astype(float)
    lo, hi = float(np.percentile(p1, 1)), float(p1.max())
    disp = np.clip((p1 - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    t = d['tracks']
    T, H, W = d['calcium'].shape
    return clean({
        'description': 'Synthetic dual-channel movie made by piezo1_magik/data/demo_dual_channel.py: PIEZO1 channel '
                       'from the repo generator (synthetic_generator.py), calcium channel = noise plus calcium events '
                       'placed on tracked puncta. Not a measurement.',
        'shape': [T, H, W], 'pixel_size_nm': PIXEL_SIZE_NM, 'params': {**d['params'], 'image_size': [H, W]},
        'calcium_b64': base64.b64encode(d['calcium'].astype('<u2').tobytes()).decode(),
        'piezo1_u8_b64': base64.b64encode(disp.tobytes()).decode(), 'piezo1_display_range': [lo, hi],
        'tracks': {c: t[c].to_numpy().tolist() for c in ('track_id', 'frame', 'x', 'y', 'photons')},
        'true_events': d['events'],
    })


def roi_on(movie, tracks, params=None):
    """CalciumROIAnalyzer on every track of tracks (DataFrame track_id, frame, x, y), detect_peaks options exposed."""
    p = dict(ROI_DEFAULTS, **(params or {}))
    a = CalciumROIAnalyzer(roi_size=int(p['roi_size']), baseline_method=p['baseline_method'],
                           baseline_percentile=float(p['baseline_percentile']))
    movie = np.asarray(movie, dtype=float)
    out = []
    for tid in sorted(tracks['track_id'].unique()):
        tr = tracks[tracks['track_id'] == tid]
        trace, edge = a.extract_trace_flags(movie, tr)
        dff = a.compute_delta_f_over_f(trace)
        events = a.detect_peaks(dff, threshold=float(p['threshold']), min_duration=int(p['min_duration']),
                                min_distance=int(p['min_distance']))
        out.append({'track_id': int(tid), 'baseline': a.compute_baseline(trace), 'trace': trace, 'dff': dff,
                    'edge': edge.tolist(), 'events': events})
    return clean({'params': p, 'computed_by': 'python', 'tracks': out})


def run_roi(params=None):
    """ROI analysis of the demo's ground-truth tracks on its calcium channel."""
    d = demo()
    return roi_on(d['calcium'], d['tracks'], params)


def oracle_edge_probs(node_tracks, edge_index):
    """1 for candidate edges joining consecutive detections of the same ground-truth track, else 0: a perfect link
    classifier restricted to the candidate graph (the best MAGIK could do on it). Not repo code: demo helper."""
    nt = np.asarray(node_tracks)
    nxt = {}
    order = np.lexsort((np.arange(len(nt)), nt))
    for a, b in zip(order[:-1], order[1:]):
        if nt[a] == nt[b]:
            nxt[int(a)] = int(b)
    return np.array([1.0 if nxt.get(int(s)) == int(t) else 0.0 for s, t in np.asarray(edge_index).T])


def mixed_tracks(node_to_track, node_tracks):
    """Number of tracks containing detections of more than one ground-truth track (demo helper)."""
    by = {}
    for n, t in node_to_track.items():
        by.setdefault(t, set()).add(int(node_tracks[n]))
    return sum(1 for s in by.values() if len(s) > 1)


def summarize(node_to_track, nf, node_tracks):
    s = analyze_track_lengths(node_to_track, nf)
    frag = analyze_fragmentation(node_to_track, node_tracks)
    return {**s, 'fragmentation': frag, 'categories': length_categories(s['lengths']),
            'mixed': mixed_tracks(node_to_track, node_tracks), 'assignment': {str(k): v for k, v in node_to_track.items()}}


def run_tracks(params=None, tracks=None):
    """Candidate graph -> oracle links -> greedy_linking -> gap filling, on the demo's ground-truth detections."""
    p = dict(TRACK_DEFAULTS, **(params or {}))
    t = demo()['tracks'] if tracks is None else tracks
    t = t.sort_values(['frame', 'track_id'], kind='stable').reset_index(drop=True)
    nf, ei, _ = build_candidate_graph(t['frame'].to_numpy(), t['x'], t['y'], np.ones(len(t)), t['photons'],
                                      int(p['max_temporal_gap']), float(p['max_spatial_distance']))
    gt = t['track_id'].to_numpy()
    probs = oracle_edge_probs(gt, ei)
    linked = greedy_linking(probs, ei, 0.5)
    frags = build_track_fragments(linked, nf)
    merged, iterations = merge_track_fragments(frags, int(p['max_gap']), float(p['max_distance']))
    filled = fragments_to_node_mapping(merged)
    return clean({'params': p, 'computed_by': 'python', 'n_nodes': len(t), 'n_edges': int(ei.shape[1]),
                  'n_true_edges': int(probs.sum()), 'n_gt_tracks': int(len(np.unique(gt))),
                  'nodes': {'frame': nf[:, 0], 'x': nf[:, 1], 'y': nf[:, 2], 'gt': gt},
                  'linked': summarize(linked, nf, gt), 'filled': {**summarize(filled, nf, gt),
                                                                    'iterations': iterations}})
