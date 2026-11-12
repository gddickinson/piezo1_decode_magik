"""Track quality statistics (scripts/07_analyze_tracks.py, 08_fill_track_gaps.py and the web interface).

Lengths are numbers of detections, spans and gaps are in frames.
"""
from collections import defaultdict

import numpy as np


def _frames_by_track(node_to_track, node_features):
    nf = np.asarray(node_features, dtype=float)
    out = defaultdict(list)
    for node_idx, track_id in node_to_track.items():
        out[track_id].append(nf[node_idx, 0])
    return out


def analyze_track_lengths(node_to_track, node_features):
    """Per track: length (# detections), temporal span (last - first frame + 1) and gaps (missing frames inside
    the span). node_features[:, 0] is the frame."""
    lengths, spans, gaps = [], [], []
    by_track = _frames_by_track(node_to_track, node_features)
    for frames in by_track.values():
        fs = sorted(frames)
        lengths.append(len(fs))
        spans.append(fs[-1] - fs[0] + 1)
        expected = set(range(int(fs[0]), int(fs[-1]) + 1))
        gaps.append(len(expected - set(int(f) for f in fs)))
    return {'lengths': lengths, 'temporal_spans': spans, 'gaps': gaps, 'num_tracks': len(by_track)}


def analyze_gt_tracks(node_tracks, node_features):
    """Lengths and temporal spans of the ground-truth tracks (node_tracks[i] = GT track id of node i)."""
    gt = {i: t for i, t in enumerate(np.asarray(node_tracks).reshape(-1).tolist())}
    r = analyze_track_lengths(gt, node_features)
    return {'lengths': r['lengths'], 'temporal_spans': r['temporal_spans'], 'num_tracks': r['num_tracks']}


def analyze_fragmentation(pred_tracks, gt_tracks, node_features=None):
    """For each ground-truth track, the number of predicted tracks its nodes fall in (1 = intact; nodes left
    unlinked count for nothing, so a GT track with no linked node scores 0)."""
    gt_to_nodes = defaultdict(list)
    for node_idx, track_id in enumerate(np.asarray(gt_tracks).reshape(-1).tolist()):
        gt_to_nodes[track_id].append(node_idx)
    return [len({pred_tracks[n] for n in nodes if n in pred_tracks}) for nodes in gt_to_nodes.values()]


def length_categories(lengths):
    """Counts of short (<= 3), medium (4-10) and long (> 10 detections) tracks, as in 07_analyze_tracks.py."""
    return {'short': sum(1 for n in lengths if n <= 3), 'medium': sum(1 for n in lengths if 3 < n <= 10),
            'long': sum(1 for n in lengths if n > 10)}
