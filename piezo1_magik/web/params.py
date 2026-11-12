"""Parameters of the pipeline, read from the repo itself: configs/*.yaml (values and inline comments) and the
argparse options of every script in scripts/ (parsed with ast, nothing is imported or run). Units come from the
code's own comments/help where it gives them (UNITS below); 'px' values are also shown in um at 130 nm/px."""
import ast
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PIXEL_SIZE_NM = 130.0  # psf_models default and the comment in configs/magik_training.yaml

# unit for a parameter name (YAML key or CLI option); source in the comment
UNITS = {
    'max_spatial_distance': 'px',   # magik_training.yaml: "pixels (at 130nm/px = 3.9 um)"
    'max_temporal_gap': 'frames',   # "Link across up to 2 frames"
    'max_frame_gap': 'frames', 'max_gap': 'frames',  # pipeline.yaml / 08 help "Maximum frame gap to bridge"
    'max_distance': 'px',           # pipeline.yaml "(pixels)", 04/08 help "(pixels)"
    'roi_size': 'px (side of square ROI)',  # "ROI size (5x5 pixels)"
    'threshold': 'probability 0-1', 'detection_threshold': 'probability 0-1', 'link_threshold': 'probability 0-1',
    'event_threshold': 'x robust sigma above median dF/F0',  # detect_peaks docstring
    'min_duration': 'frames', 'min_distance': 'frames', 'viz_track_length': 'frames',
    'baseline_percentile': 'percentile',
    'min_photons': 'photons', 'photons_mean': 'photons', 'photons_std': 'photons',
    'baseline': 'ADU', 'read_noise': 'electrons (added to ADU 1:1)',  # 01 help; add_noise adds it directly
    'image_size': 'px', 'num_frames': 'frames',
}

CONFIG_READERS = {
    'decode_training.yaml': 'passed with --config to scripts/02_train_decode.py (and 02b_resume_decode_training.py)',
    'magik_training.yaml': 'passed with --config to scripts/03_train_magik.py; 03b resumes from the config stored '
                           'in the checkpoint',
    'pipeline.yaml': 'not read by any script: 04_run_pipeline.py and 09_run_complete_pipeline.py take command-line '
                     'options instead (listed below)',
}


def _flatten(d, prefix=''):
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            yield from _flatten(v, key + '.')
        else:
            yield key, v


def _comments(text):
    """{key: inline comment} for 'key: value  # comment' lines."""
    out = {}
    for line in text.splitlines():
        m = re.match(r'\s*([\w]+):\s*[^#]*#\s*(.+)$', line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def px_to_um(value, unit):
    if unit.startswith('px') and isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(value * PIXEL_SIZE_NM / 1000, 4)
    if unit.startswith('px') and isinstance(value, list):
        return [round(v * PIXEL_SIZE_NM / 1000, 4) for v in value]
    return None


def configs():
    out = []
    for path in sorted((ROOT / 'configs').glob('*.yaml')):
        text = path.read_text()
        comments = _comments(text)
        rows = []
        for key, value in _flatten(yaml.safe_load(text) or {}):
            leaf = key.split('.')[-1]
            unit = UNITS.get(leaf, '')
            rows.append({'key': key, 'value': value, 'unit': unit, 'um': px_to_um(value, unit),
                         'comment': comments.get(leaf, '')})
        out.append({'file': f'configs/{path.name}', 'read_by': CONFIG_READERS.get(path.name, ''), 'rows': rows})
    return out


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def script_options():
    out = []
    for path in sorted((ROOT / 'scripts').glob('[0-9]*.py')):
        tree = ast.parse(path.read_text())
        doc = (ast.get_docstring(tree) or '').strip().splitlines()
        opts = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' and node.args):
                name = _literal(node.args[0])
                kw = {k.arg: k.value for k in node.keywords}
                leaf = str(name).lstrip('-')
                action = _literal(kw['action']) if 'action' in kw else None
                default = _literal(kw['default']) if 'default' in kw else (False if action == 'store_true' else None)
                unit = UNITS.get(leaf, '')
                opts.append({'name': name, 'type': (ast.unparse(kw['type']) if 'type' in kw else
                                                    ('flag' if action else 'str')),
                             'default': default, 'required': bool(_literal(kw['required'])) if 'required' in kw else False,
                             'help': _literal(kw['help']) if 'help' in kw else '', 'unit': unit,
                             'um': px_to_um(default, unit)})
        if opts:
            out.append({'script': f'scripts/{path.name}', 'summary': doc[0] if doc else '', 'options': opts})
    return out


def all_params():
    c, s = configs(), script_options()
    return {'pixel_size_nm': PIXEL_SIZE_NM, 'configs': c, 'scripts': s,
            'n_config_values': sum(len(x['rows']) for x in c), 'n_options': sum(len(x['options']) for x in s)}
