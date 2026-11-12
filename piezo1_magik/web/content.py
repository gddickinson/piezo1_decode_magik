"""Pipeline description for the web pages, written from the README and the code (file references included so every
statement can be checked). Nothing here is a measured result."""

STAGES = [
    {
        'id': 'decode', 'n': 1, 'title': 'DECODE localization', 'where': 'local Python only (PyTorch + trained checkpoint)',
        'code': ['piezo1_magik/models/decode_net.py', 'scripts/02_train_decode.py', 'scripts/02c_evaluate_decode.py'],
        'input': 'PIEZO1 channel, (T, H, W); 04_run_pipeline.py divides by 65535 and slides a 3-frame window',
        'output': 'per detection: x, y (px, sub-pixel), photons, sigma_x, sigma_y, sigma_N, prob',
        'points': [
            'FrameAnalysisNet: a U-Net (2 poolings, base_channels = 32 in configs/decode_training.yaml) applied to each of '
            'the 3 frames separately.',
            'TemporalContextNet: a second U-Net on the 3 x base_channels concatenated features.',
            'Four 1x1-conv heads: probability (sigmoid), sub-pixel offset (tanh x 0.5 px), photons (softplus) and '
            '3 uncertainties (softplus).',
            'DECODENet.predict keeps every pixel whose probability exceeds the threshold and adds its offset: '
            'x = column + offset_x, y = row + offset_y (pixel-centre convention).',
            '04_run_pipeline.py runs it on frames 1 to T-2 (the first and last frame have no 3-frame window).',
        ],
    },
    {
        'id': 'magik', 'n': 2, 'title': 'MAGIK graph tracking', 'where': 'local Python only (PyTorch + trained checkpoint); '
        'the graph building, linking and gap filling around it also run in this site',
        'code': ['piezo1_magik/models/magik_gnn.py', 'piezo1_magik/tracking/linking.py', 'scripts/03_train_magik.py',
                 'scripts/09_run_complete_pipeline.py'],
        'input': 'all detections of a movie',
        'output': 'track_id per detection (tracks.csv, track_summary.csv in 09)',
        'points': [
            'Graph: one node per detection with features [frame, x, y, prob, photons]; a candidate edge from each node to '
            'every detection 1 to max_temporal_gap frames later within max_spatial_distance px, with features '
            '[dx, dy, distance, temporal_gap] (training config: 2 frames, 30 px = 3.9 um at 130 nm/px).',
            'MAGIKNet: node and edge MLP encoders (hidden_dim 64), 3 EdgeConv message-passing layers (sum aggregation, '
            'residual), and an edge classifier giving one logit per candidate edge; sigmoid gives the link probability.',
            'Linking in 03c/07/08/09: greedy, highest probability first, each detection with at most one link in and one '
            'out (greedy_linking). 04_run_pipeline.py instead joins all edges above --link_threshold into connected '
            'components.',
            'Optional gap filling (09 --use_gap_filling, 08): a fragment is extended by another that starts at most '
            'max_gap frames after it ends and within max_distance px of its linearly extrapolated position; best '
            'score (distance + 2 x gap) first.',
        ],
    },
    {
        'id': 'roi', 'n': 3, 'title': 'ROI calcium analysis', 'where': 'runs in this site (JS port) and in the local app (Python)',
        'code': ['piezo1_magik/analysis/calcium_roi.py', 'scripts/04_run_pipeline.py'],
        'input': 'calcium channel (T, H, W) and the tracks (track_id, frame, x, y)',
        'output': 'calcium_traces.csv (F, dF/F0, F0 per track and frame), calcium_events.csv',
        'points': [
            'F(t): mean of a roi_size x roi_size pixel square (default 5 x 5 = 0.65 x 0.65 um) centred on the rounded '
            'punctum position; pixels outside the image are left out of the mean and the frame is flagged as an edge '
            'ROI (edge_roi in calcium_traces.csv); frames where the track has no detection are NaN.',
            'F0: the 10th percentile of F (baseline_method "percentile"); "rolling" uses the median of a centred rolling '
            'minimum (window min(50, n/5) frames); anything else, the mean.',
            'dF/F0 = (F - F0) / F0.',
            'Events: scipy.signal.find_peaks on dF/F0 with height = threshold x std(dF/F0) (default 2) and distance = '
            'min_distance frames (default 5); each event extends while dF/F0 stays above half that height and is kept '
            'if it lasts at least min_duration frames (default 2).',
        ],
    },
]

FACTS = {
    'pixel_size_nm': 130.0,
    'frame_interval': 'not defined anywhere in the repo, so times are given in frames',
    'citations': ['DECODE: Speiser et al., Nature Methods 2021', 'MAGIK: Jiang et al., (DeepTrack 2.0)',
                  'PIEZO1-HaloTag: Bertaccini et al., Nature Commun 2025'],
    'licence': 'MIT',
}
