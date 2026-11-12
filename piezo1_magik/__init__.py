"""PIEZO1 DECODE-MAGIK: DECODE localization + MAGIK GNN tracking + ROI calcium analysis.

Subpackages:
    models    -- DECODENet (localization U-Nets) and MAGIKNet (tracking GNN); need PyTorch
    data      -- synthetic movie generator and PyTorch datasets
    analysis  -- CalciumROIAnalyzer (ROI traces, dF/F0, event detection)
    utils     -- PSF models and camera noise
See INTERFACE.md for the full navigation map.
"""
