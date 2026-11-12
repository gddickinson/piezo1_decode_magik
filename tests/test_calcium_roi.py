"""Tests for CalciumROIAnalyzer (piezo1_magik/analysis/calcium_roi.py)."""
import numpy as np
import pandas as pd

from piezo1_magik.analysis.calcium_roi import CalciumROIAnalyzer


def test_extract_roi_inside_image():
    img = np.arange(100, dtype=float).reshape(10, 10)
    roi = CalciumROIAnalyzer(roi_size=3).extract_roi(img, x=4.2, y=6.0)
    np.testing.assert_array_equal(roi, img[5:8, 3:6])


def test_extract_roi_marks_out_of_frame_pixels_nan():
    """Pixels outside the image are NaN (not zeros); a ROI entirely outside is all NaN. Regression for the
    negative-slice wrap-around: a centre well left of/above the image must not pick up pixels from across it."""
    img = np.ones((10, 10))
    a = CalciumROIAnalyzer(roi_size=5)
    for x, y in [(-10.0, 5.0), (5.0, -10.0), (-4.0, -4.0), (30.0, 5.0)]:
        roi = a.extract_roi(img, x, y)
        assert roi.shape == (5, 5) and np.isnan(roi).all(), (x, y)
    roi = a.extract_roi(img, 0.0, 9.0)  # corner: 3 x 3 of the 5 x 5 inside
    assert np.isfinite(roi).sum() == 9 and np.nanmean(roi) == 1.0


def test_edge_roi_mean_uses_valid_pixels_and_is_flagged():
    img = np.full((10, 10), 7.0)
    a = CalciumROIAnalyzer(roi_size=5)
    assert a.roi_mean(img, 5.0, 5.0) == (7.0, 25)
    assert a.roi_mean(img, 0.0, 5.0) == (7.0, 15)
    m, n = a.roi_mean(img, -10.0, 5.0)
    assert np.isnan(m) and n == 0
    movie = np.full((3, 10, 10), 7.0)
    track = pd.DataFrame({'frame': [0, 1, 2], 'x': [5.0, 2.0, 0.0], 'y': [5.0, 5.0, 5.0]})
    trace, edge = a.extract_trace_flags(movie, track)
    assert trace.tolist() == [7.0, 7.0, 7.0] and edge.tolist() == [False, False, True]


def test_no_false_events_when_a_track_reaches_the_image_edge():
    """Regression: zero-padding the part of the ROI outside the image lowered F at the border, which lowered F0
    (a low percentile) and turned the interior frames into 'events' on a movie with no calcium signal at all."""
    rng = np.random.default_rng(1)
    movie = rng.normal(500, 5, size=(60, 30, 30))  # flat: no events anywhere
    xs = np.r_[np.full(30, 15.0), np.linspace(15, 0, 10), np.zeros(20)]  # moves onto the left border
    track = pd.DataFrame({'frame': range(60), 'x': xs, 'y': np.full(60, 15.0)})
    r = CalciumROIAnalyzer(roi_size=5).analyze_track(movie, track, track_id=0)
    f = r['trace']
    assert abs(np.mean(f[-20:]) - np.mean(f[:30])) < 2  # edge frames not biased (zero padding gave -40%)
    assert all(abs(e['amplitude']) < 0.02 for e in r['events']), r['events']  # was 0.67 with zero padding
    assert r['edge'][-20:].all() and not r['edge'][:30].any()


def test_trace_dff_and_peaks():
    rng = np.random.default_rng(0)
    movie = rng.normal(500, 5, size=(60, 20, 20))
    movie[30:36, 8:13, 8:13] += 300
    track = pd.DataFrame({'frame': range(60), 'x': [10.0] * 60, 'y': [10.0] * 60})
    a = CalciumROIAnalyzer(roi_size=5)
    r = a.analyze_track(movie, track, track_id=0)
    assert abs(r['baseline'] - 500) < 5
    assert [e['peak_frame'] for e in r['events']] and all(30 <= e['peak_frame'] <= 35 for e in r['events'])


def test_even_roi_size_rejected_up_front():
    """Regression: an even roi_size (e.g. 4) cut roi_size + 1 pixels and then crashed with a NumPy broadcast error
    inside extract_roi on the first punctum. It is now rejected when the analyzer is created."""
    import pytest
    with pytest.raises(ValueError, match="odd"):
        CalciumROIAnalyzer(roi_size=4)
    with pytest.raises(ValueError):
        CalciumROIAnalyzer(roi_size=0)


def test_event_threshold_measured_from_the_trace_baseline():
    """Regression: the threshold was threshold x std(dF/F0) measured from zero, but dF/F0 sits above zero
    (F0 is a low percentile), so noise crossed it. It is now median + threshold x robust sigma (1.4826 x MAD)."""
    rng = np.random.default_rng(4)
    dff = 0.1 + rng.normal(0, 0.01, 400)  # offset noise, no events
    a = CalciumROIAnalyzer(roi_size=5)
    assert a.detect_peaks(dff, threshold=3.0) == []
    dff[200:206] += 0.2  # one real event
    ev = a.detect_peaks(dff, threshold=3.0)
    assert len(ev) == 1 and 200 <= ev[0]['peak_frame'] <= 205


def test_demo_noise_events_mostly_rejected():
    """On the synthetic demo (4 placed events) nearly all detections are placed events."""
    from piezo1_magik.web.demo import run_roi, demo
    placed = demo()['events']
    out = run_roi()
    hits = false = 0
    for t in out['tracks']:
        for e in t['events']:
            real = any(p['track_id'] == t['track_id'] and p['start_frame'] <= e['peak_frame'] <= p['end_frame']
                       for p in placed)
            hits += real
            false += not real
    assert hits >= 3 and false <= 2, (hits, false)  # was 4 hits and 16 false with the zero-based threshold


def test_no_events_at_all_when_a_track_reaches_the_image_edge():
    rng = np.random.default_rng(1)
    movie = rng.normal(500, 5, size=(60, 30, 30))
    xs = np.r_[np.full(30, 15.0), np.linspace(15, 0, 10), np.zeros(20)]
    track = pd.DataFrame({'frame': range(60), 'x': xs, 'y': np.full(60, 15.0)})
    r = CalciumROIAnalyzer(roi_size=5).analyze_track(movie, track, track_id=0)
    assert r['events'] == [], r['events']
