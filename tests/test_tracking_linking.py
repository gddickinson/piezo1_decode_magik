"""Tests for piezo1_magik.tracking (pure NumPy linking, gap filling and track statistics)."""
import numpy as np

from piezo1_magik.tracking.linking import (build_candidate_graph, build_track_fragments, fragments_to_node_mapping,
                                          greedy_linking, merge_track_fragments)
from piezo1_magik.tracking.track_stats import (analyze_fragmentation, analyze_track_lengths, length_categories)


def test_candidate_graph_respects_gap_and_distance():
    frames, xs, ys = [0, 0, 1, 3], [0.0, 10.0, 1.0, 2.0], [0.0, 0.0, 0.0, 0.0]
    nf, ei, ef = build_candidate_graph(frames, xs, ys, [1] * 4, [1000] * 4, max_temporal_gap=2, max_spatial_distance=5)
    assert nf.shape == (4, 5)
    assert ei.T.tolist() == [[0, 2], [2, 3]]  # 0->3 is 3 frames apart, 1->2 is 9 px apart
    np.testing.assert_allclose(ef[1], [1.0, 0.0, 1.0, 2.0])


def test_greedy_linking_one_to_one_highest_first():
    ei = np.array([[0, 0, 1], [2, 3, 3]])
    probs = np.array([0.9, 0.8, 0.95])
    n2t = greedy_linking(probs, ei, threshold=0.5)
    # 1->3 (0.95) first, then 0->2 (0.9); 0->3 is rejected because 3 already has an incoming link
    assert n2t[1] == n2t[3] and n2t[0] == n2t[2] and n2t[0] != n2t[1]
    assert greedy_linking(probs, ei, threshold=0.99) == {}


def test_gap_filling_bridges_a_blink_and_stats():
    # one particle moving +1 px/frame, missing frames 3-4, linked as two fragments
    frames = [0, 1, 2, 5, 6, 7]
    nf = np.array([[f, f * 1.0, 5.0, 1, 1000] for f in frames])
    n2t = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    frags = build_track_fragments(n2t, nf)
    merged, _ = merge_track_fragments(frags, max_gap=5, max_distance=1.0)
    assert len(merged) == 1
    m = fragments_to_node_mapping(merged)
    s = analyze_track_lengths(m, nf)
    assert s == {'lengths': [6], 'temporal_spans': [8.0], 'gaps': [2], 'num_tracks': 1}
    assert analyze_fragmentation(n2t, [0] * 6) == [2]
    assert analyze_fragmentation(m, [0] * 6) == [1]
    assert length_categories([1, 3, 4, 10, 11]) == {'short': 2, 'medium': 2, 'long': 1}
