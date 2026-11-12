"""Candidate graph, greedy linking and gap filling for MAGIK tracking (NumPy only).

Shared by scripts/03c_evaluate_magik.py, 07_analyze_tracks.py, 08_fill_track_gaps.py and
09_run_complete_pipeline.py (previously each held its own copy) and by the web interface. Positions are in
pixels and times in frames throughout; inputs may be NumPy arrays or CPU torch tensors.
"""
from collections import defaultdict

import numpy as np


def build_candidate_graph(frames, xs, ys, probs, photons, max_temporal_gap, max_spatial_distance):
    """Candidate edges between detections, as fed to MAGIK (scripts/09_run_complete_pipeline.py).

    Node i links to node j > i when 0 < frame_j - frame_i <= max_temporal_gap (frames) and the distance
    is <= max_spatial_distance (pixels). Detections are expected in frame order.

    Returns:
        node_features: (N, 5) [frame, x, y, prob, photons]
        edge_index: (2, E) int64 [source, target]
        edge_features: (E, 4) float32 [dx, dy, distance, temporal_gap]
    """
    frames = np.asarray(frames)
    positions = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    node_features = np.column_stack([frames, positions, np.asarray(probs), np.asarray(photons)])
    edges, feats = [], []
    n = len(frames)
    for i in range(n):
        for j in range(i + 1, n):
            gap = frames[j] - frames[i]
            if gap <= 0 or gap > max_temporal_gap:
                continue
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance <= max_spatial_distance:
                edges.append([i, j])
                feats.append([dx, dy, distance, gap])
    if not edges:
        return node_features, np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    return node_features, np.array(edges, dtype=np.int64).T, np.array(feats, dtype=np.float32)


def greedy_linking(edge_probs, edge_index, threshold=0.5):
    """Greedy one-to-one linking: take edges above threshold, highest probability first, keeping each node to at
    most one outgoing and one incoming link; follow the links into tracks.

    Returns:
        node_to_track: {node index: track id}. Nodes with no accepted link are not included.
    """
    probs = np.asarray(edge_probs, dtype=float).reshape(-1)
    index = np.asarray(edge_index)
    mask = probs > threshold
    if mask.sum() == 0:
        return {}
    valid_edges = index[:, mask]
    valid_probs = probs[mask]
    sorted_idx = np.argsort(-valid_probs, kind="stable")

    src_linked, dst_linked, links = set(), set(), []
    for idx in sorted_idx:
        src, dst = int(valid_edges[0, idx]), int(valid_edges[1, idx])
        if src not in src_linked and dst not in dst_linked:
            links.append((src, dst))
            src_linked.add(src)
            dst_linked.add(dst)

    forward_links, backward_links = {}, {}
    for src, dst in links:
        forward_links[src] = dst
        backward_links[dst] = src

    node_to_track, track_counter = {}, 0
    all_nodes = set(forward_links) | set(backward_links)
    for start_node in [n for n in all_nodes if n not in backward_links]:
        if start_node in node_to_track:
            continue
        current = start_node
        while current is not None:
            node_to_track[current] = track_counter
            current = forward_links.get(current, None)
        track_counter += 1
    for node in all_nodes:
        if node not in node_to_track:
            node_to_track[node] = track_counter
            track_counter += 1
    return node_to_track


def build_track_fragments(node_to_track, node_features):
    """Group nodes by track into fragments sorted by frame.

    Each fragment: track_id, nodes, frames, positions [(x, y)], start_frame, end_frame, start_pos, end_pos, length.
    """
    nf = np.asarray(node_features, dtype=float)
    track_to_nodes = defaultdict(list)
    for node_idx, track_id in node_to_track.items():
        track_to_nodes[track_id].append(node_idx)
    fragments = []
    for track_id, node_indices in track_to_nodes.items():
        data = sorted(((nf[n, 0], nf[n, 1], nf[n, 2], n) for n in node_indices), key=lambda d: d[0])
        frames = [d[0] for d in data]
        positions = [(d[1], d[2]) for d in data]
        fragments.append({
            'track_id': track_id, 'nodes': [d[3] for d in data], 'frames': frames, 'positions': positions,
            'start_frame': frames[0], 'end_frame': frames[-1], 'start_pos': positions[0], 'end_pos': positions[-1],
            'length': len(data),
        })
    return fragments


def estimate_velocity(fragment):
    """Mean velocity (pixels/frame) from the fragment's first and last positions; (0, 0) for a single point."""
    if len(fragment['frames']) < 2:
        return (0, 0)
    (x0, y0), (x1, y1) = fragment['start_pos'], fragment['end_pos']
    dt = fragment['end_frame'] - fragment['start_frame']
    if dt == 0:
        return (0, 0)
    return ((x1 - x0) / dt, (y1 - y0) / dt)


def predict_position(fragment, target_frame):
    """Linear extrapolation of the fragment's end position to target_frame (pixels)."""
    vx, vy = estimate_velocity(fragment)
    x0, y0 = fragment['end_pos']
    dt = target_frame - fragment['end_frame']
    return (x0 + vx * dt, y0 + vy * dt)


def can_merge_fragments(frag1, frag2, max_gap=10, max_distance=30):
    """frag1 -> frag2 is a merge candidate when frag2 starts after frag1 ends, within max_gap frames, and starts
    within max_distance pixels of frag1's extrapolated position. Returns (ok, score = distance + 2 * gap)."""
    if frag1['end_frame'] >= frag2['start_frame']:
        return False, float('inf')
    gap = frag2['start_frame'] - frag1['end_frame']
    if gap > max_gap:
        return False, float('inf')
    px, py = predict_position(frag1, frag2['start_frame'])
    ax, ay = frag2['start_pos']
    distance = np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    if distance > max_distance:
        return False, float('inf')
    return True, distance + gap * 2


def merge_track_fragments(fragments, max_gap=10, max_distance=30, max_iterations=None):
    """Gap filling: repeatedly merge the best-scoring candidate pair (lowest distance + 2 * gap) until none is left.

    Each merge removes one fragment, so the loop ends after at most len(fragments) passes; max_iterations
    (default: no extra cap) can stop it earlier. (The scripts used to stop at 100 merges.)

    Returns:
        (final_fragments, n_iterations)
    """
    active = {i: frag.copy() for i, frag in enumerate(fragments)}
    merged_into = {}
    iteration = 0
    if max_iterations is None:
        max_iterations = len(fragments) + 1
    while iteration < max_iterations:
        iteration += 1
        candidates = []
        for i, frag1 in active.items():
            if i in merged_into:
                continue
            for j, frag2 in active.items():
                if j in merged_into or i == j:
                    continue
                ok, quality = can_merge_fragments(frag1, frag2, max_gap, max_distance)
                if ok:
                    candidates.append((i, j, quality))
        if not candidates:
            break
        candidates.sort(key=lambda c: c[2])
        i, j, _ = candidates[0]
        f1, f2 = active[i], active[j]
        active[i] = {
            'track_id': f1['track_id'], 'nodes': f1['nodes'] + f2['nodes'], 'frames': f1['frames'] + f2['frames'],
            'positions': f1['positions'] + f2['positions'], 'start_frame': f1['start_frame'],
            'end_frame': f2['end_frame'], 'start_pos': f1['start_pos'], 'end_pos': f2['end_pos'],
            'length': f1['length'] + f2['length'],
        }
        merged_into[j] = i
    return [frag for i, frag in active.items() if i not in merged_into], iteration


def fragments_to_node_mapping(fragments):
    """{node index: track id} with track ids renumbered 0..n-1 in fragment order."""
    return {node: track_id for track_id, frag in enumerate(fragments) for node in frag['nodes']}
