"""configs/pipeline.yaml for scripts/04_run_pipeline.py: load it and apply command-line overrides.

Every key in the YAML maps to one setting below (OPTION_KEYS); a command-line option given explicitly wins over the
file. Units: distances in pixels, gaps/durations in frames, thresholds as documented in the YAML comments.
"""
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "pipeline.yaml"

# command-line option name -> (section, key) in pipeline.yaml
OPTION_KEYS = {
    'decode': ('models', 'decode'),
    'magik': ('models', 'magik'),
    'detection_threshold': ('detection', 'threshold'),
    'min_photons': ('detection', 'min_photons'),
    'max_frame_gap': ('tracking', 'max_frame_gap'),
    'max_distance': ('tracking', 'max_distance'),
    'link_threshold': ('tracking', 'link_threshold'),
    'roi_size': ('calcium', 'roi_size'),
    'baseline_method': ('calcium', 'baseline_method'),
    'baseline_percentile': ('calcium', 'baseline_percentile'),
    'event_threshold': ('calcium', 'event_threshold'),
    'min_duration': ('calcium', 'min_duration'),
    'min_distance': ('calcium', 'min_distance'),
    'save_tracks': ('output', 'save_tracks'),
    'save_calcium_traces': ('output', 'save_calcium_traces'),
    'save_events': ('output', 'save_events'),
    'device': ('processing', 'device'),
}


def load_pipeline_config(path=DEFAULT_CONFIG) -> dict:
    """Flat {option: value} from the YAML. Raises if the file has keys no option uses, or lacks one."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    present = {(sec, key) for sec, d in raw.items() if isinstance(d, dict) for key in d}
    unknown = present - set(OPTION_KEYS.values())
    if unknown:
        raise ValueError(f"{path}: keys not used by 04_run_pipeline.py: {sorted('.'.join(k) for k in unknown)}")
    missing = set(OPTION_KEYS.values()) - present
    if missing:
        raise ValueError(f"{path}: missing keys: {sorted('.'.join(k) for k in missing)}")
    return {opt: raw[sec][key] for opt, (sec, key) in OPTION_KEYS.items()}


def resolve(cli: dict, path=DEFAULT_CONFIG) -> dict:
    """Config values, overridden by every command-line value that is not None."""
    cfg = load_pipeline_config(path)
    cfg.update({k: v for k, v in cli.items() if v is not None})
    return cfg
