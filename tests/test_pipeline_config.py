"""configs/pipeline.yaml is read by scripts/04_run_pipeline.py, with command-line overrides.

Regression: the file existed but no script read it, so editing it changed nothing.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from piezo1_magik.analysis.calcium_roi import analyze_all_tracks
from piezo1_magik.pipeline_config import DEFAULT_CONFIG, load_pipeline_config, resolve

ROOT = Path(__file__).resolve().parent.parent


def test_every_yaml_key_is_used_and_loaded():
    cfg = load_pipeline_config(DEFAULT_CONFIG)
    assert cfg['max_frame_gap'] == 2 and cfg['roi_size'] == 5 and cfg['event_threshold'] == 2.0
    assert cfg['save_tracks'] is True and cfg['device'] == 'cuda'


def test_unknown_or_missing_keys_rejected(tmp_path):
    text = DEFAULT_CONFIG.read_text()
    (tmp_path / "a.yaml").write_text(text + "\nextra:\n  unused: 1\n")
    with pytest.raises(ValueError, match="not used"):
        load_pipeline_config(tmp_path / "a.yaml")
    (tmp_path / "b.yaml").write_text(text.replace("  min_photons: 100", "  # removed"))
    with pytest.raises(ValueError, match="missing"):
        load_pipeline_config(tmp_path / "b.yaml")


def test_command_line_overrides_config():
    cfg = resolve({'roi_size': 7, 'max_distance': None, 'save_events': False})
    assert cfg['roi_size'] == 7 and cfg['max_distance'] == 10.0 and cfg['save_events'] is False


def test_04_parser_reads_config(tmp_path):
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("run_pipeline_04", ROOT / "scripts" / "04_run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.build_parser().parse_args(['--input', 'm.tif', '--event_threshold', '3.5', '--no-save_tracks'])
    assert args.config == str(DEFAULT_CONFIG)
    cli = {k: v for k, v in vars(args).items() if k not in ('config', 'input', 'output', 'gpu')}
    cfg = resolve(cli, args.config)
    assert cfg['event_threshold'] == 3.5 and cfg['save_tracks'] is False and cfg['link_threshold'] == 0.5
    dets = [[{'photons': 50.0}, {'photons': 150.0}]]
    assert mod.filter_min_photons(dets, cfg['min_photons']) == [[{'photons': 150.0}]]


def test_calcium_options_reach_event_detection():
    rng = np.random.default_rng(0)
    movie = rng.normal(500, 5, size=(60, 20, 20))
    movie[30:33, 8:13, 8:13] += 300
    tracks = pd.DataFrame({'track_id': 0, 'frame': range(60), 'x': 10.0, 'y': 10.0})
    _, ev = analyze_all_tracks(movie, tracks, roi_size=5)
    assert len(ev) == 1
    _, ev = analyze_all_tracks(movie, tracks, roi_size=5, event_threshold=1e6)
    assert len(ev) == 0


def test_max_frame_gap_default_matches_magik_training_gap():
    """MAGIK sees the temporal gap as an edge feature; 04 must not build edges spanning more frames than in training."""
    import yaml
    train = yaml.safe_load((ROOT / "configs" / "magik_training.yaml").read_text())
    assert load_pipeline_config(DEFAULT_CONFIG)['max_frame_gap'] == train['data']['max_temporal_gap'] == 2


def _run_04_main(monkeypatch, tmp_path, argv):
    """Run 04's main() with the models, movie I/O and DECODE stubbed out; return the max_frame_gap values that
    reached build_candidate_graph and the candidate edges' temporal gaps."""
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("run_pipeline_04", ROOT / "scripts" / "04_run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Net(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def load_state_dict(self, *a, **k):
            return None

    class Magik(mod.MAGIKNet):
        def load_state_dict(self, *a, **k):
            return None

    seen = []

    def spy_graph(frames, xs, ys, probs, photons, max_frame_gap, max_distance):
        nf, ei, ef = real_graph(frames, xs, ys, probs, photons, max_frame_gap, max_distance)
        seen.append((max_frame_gap, sorted({int(g) for g in ef[:, 3]}) if len(ef) else []))
        return nf, ei, ef

    real_graph = mod.build_candidate_graph
    det = {"x": 10.0, "y": 10.0, "photons": 500.0, "sigma_x": .1, "sigma_y": .1, "prob": .9}
    # one punctum, seen in frames 0, 1, 4, 9: frame gaps 1, 3, 5
    dets = [[dict(det)] if t in (0, 1, 4, 9) else [] for t in range(10)]
    monkeypatch.setattr(mod.torch, "load", lambda *a, **k: {"model_state_dict": {}})
    monkeypatch.setattr(mod, "DECODENet", Net)
    monkeypatch.setattr(mod, "MAGIKNet", Magik)
    monkeypatch.setattr(mod, "load_dual_channel_movie", lambda p: (np.zeros((10, 16, 16)), None))
    monkeypatch.setattr(mod, "detect_puncta_decode", lambda *a, **k: dets)
    monkeypatch.setattr(mod, "build_candidate_graph", spy_graph)
    monkeypatch.setattr(mod, "save_results", lambda *a, **k: None)
    mod.main(['--input', 'm.tif', '--output', str(tmp_path), '--device', 'cpu', *argv])
    return seen


def test_max_frame_gap_from_config_reaches_the_linker(monkeypatch, tmp_path):
    """The tracking.max_frame_gap value (and a --max_frame_gap override) is what 04 builds MAGIK's graph with."""
    assert _run_04_main(monkeypatch, tmp_path, []) == [(2, [1])]  # default config: the 3- and 5-frame gaps are not edges
    cfg = tmp_path / "gap5.yaml"
    cfg.write_text(DEFAULT_CONFIG.read_text().replace("max_frame_gap: 2", "max_frame_gap: 5"))
    assert _run_04_main(monkeypatch, tmp_path, ['--config', str(cfg)]) == [(5, [1, 3, 4, 5])]
    assert _run_04_main(monkeypatch, tmp_path, ['--config', str(cfg), '--max_frame_gap', '3']) == [(3, [1, 3])]


def test_noise_window_from_config_and_cli_reaches_event_detection(monkeypatch):
    """calcium.noise_window (default 61) and --noise_window reach detect_peaks through analyze_all_tracks."""
    import piezo1_magik.analysis.calcium_roi as roi
    assert load_pipeline_config(DEFAULT_CONFIG)['noise_window'] == 61
    seen = []
    real = roi.CalciumROIAnalyzer.detect_peaks

    def spy(self, dff, **kw):
        seen.append(kw['noise_window'])
        return real(self, dff, **kw)

    monkeypatch.setattr(roi.CalciumROIAnalyzer, 'detect_peaks', spy)
    movie = np.random.default_rng(0).normal(500, 5, size=(20, 12, 12))
    tracks = pd.DataFrame({'track_id': 0, 'frame': range(20), 'x': 6.0, 'y': 6.0})
    keys = ('roi_size', 'baseline_method', 'baseline_percentile', 'event_threshold', 'min_duration', 'min_distance',
            'noise_window')
    for cli, want in (({}, 61), ({'noise_window': 0}, 0), ({'noise_window': 15}, 15)):
        cfg = resolve(cli)
        roi.analyze_all_tracks(movie, tracks, **{k: cfg[k] for k in keys})
        assert seen[-1] == want


def test_04_parser_has_noise_window():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("run_pipeline_04", ROOT / "scripts" / "04_run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build_parser().parse_args(['--input', 'm.tif', '--noise_window', '0']).noise_window == 0
    assert mod.build_parser().parse_args(['--input', 'm.tif']).noise_window is None  # None: take the config value
