"""Tests for CalciumROIAnalyzer (piezo1_magik/analysis/calcium_roi.py)."""
import numpy as np
import pandas as pd

from piezo1_magik.analysis.calcium_roi import CalciumROIAnalyzer


def test_extract_roi_inside_image():
    img = np.arange(100, dtype=float).reshape(10, 10)
    roi = CalciumROIAnalyzer(roi_size=3).extract_roi(img, x=4.2, y=6.0)
    np.testing.assert_array_equal(roi, img[5:8, 3:6])


def test_extract_roi_fully_outside_returns_zeros():
    """Regression: a centre well left of/above the image gave a negative slice end, which NumPy reads as 'count from
    the end', so the ROI picked up unrelated pixels from across the image (or, for an image wider than 2 * roi_size,
    raised ValueError when padding). The docstring promises zeros."""
    img = np.ones((10, 10))
    a = CalciumROIAnalyzer(roi_size=5)
    for x, y in [(-10.0, 5.0), (5.0, -10.0), (-4.0, -4.0), (30.0, 5.0)]:
        roi = a.extract_roi(img, x, y)
        assert roi.shape == (5, 5) and not roi.any(), (x, y)


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
