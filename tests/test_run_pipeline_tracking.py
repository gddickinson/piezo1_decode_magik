"""Regression test: scripts/04_run_pipeline.py stage 2 (track_with_magik) must run on DECODE-style detections.

It used to call build_tracking_graph() with keyword arguments it does not take and MAGIKNet.predict_links(),
which does not exist, so every run of the pipeline stopped at stage 2.
"""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location("run_pipeline_04", ROOT / "scripts" / "04_run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def detections(n_frames=6):
    """Per-frame lists as returned by detect_puncta_decode (DECODENet.predict dicts, no 'frame' key)."""
    out = []
    for t in range(n_frames):
        out.append([{"x": 10.0 + 0.3 * t, "y": 12.0, "photons": 900.0, "sigma_x": 0.1, "sigma_y": 0.1,
                     "sigma_N": 5.0, "prob": 0.9},
                    {"x": 40.0, "y": 30.0 - 0.2 * t, "photons": 1100.0, "sigma_x": 0.1, "sigma_y": 0.1,
                     "sigma_N": 5.0, "prob": 0.8}])
    return out


def test_track_with_magik_runs_and_links():
    mod = load_script()
    torch.manual_seed(0)
    model = mod.MAGIKNet().eval()
    dets = detections()
    # threshold below any sigmoid output: every candidate edge becomes a link
    tracks = mod.track_with_magik(dets, model, "cpu", max_frame_gap=1, max_distance=5.0, link_threshold=-1.0)
    assert len(tracks) == 12
    assert tracks["track_id"].nunique() == 2
    for _, g in tracks.groupby("track_id"):
        assert sorted(g["frame"]) == list(range(6))
        assert g["x"].std() < 5  # the two particles (30 px apart) are not mixed
    # threshold above any sigmoid output: no links, every detection its own track
    none = mod.track_with_magik(dets, model, "cpu", max_frame_gap=1, max_distance=5.0, link_threshold=1.0)
    assert none["track_id"].nunique() == 12
