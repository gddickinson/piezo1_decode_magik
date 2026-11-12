"""PIEZO1 DECODE-MAGIK: DECODE localization + MAGIK GNN tracking + ROI calcium analysis.

Subpackages:
    models    -- DECODENet (localization U-Nets) and MAGIKNet (tracking GNN); need PyTorch
    data      -- synthetic movie generator and PyTorch datasets
    analysis  -- CalciumROIAnalyzer (ROI traces, dF/F0, event detection)
    utils     -- PSF models and camera noise
    tracking  -- linking, gap filling, track statistics (NumPy only)
    pipeline_config -- configs/pipeline.yaml loader for scripts/04_run_pipeline.py
    web       -- web interface
See INTERFACE.md for the full navigation map.
"""
