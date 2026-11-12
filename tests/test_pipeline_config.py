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
    assert cfg['max_frame_gap'] == 5 and cfg['roi_size'] == 5 and cfg['event_threshold'] == 2.0
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
