# PIEZO1 DECODE-MAGIK Tracker -- Roadmap

## Current State
A modular pipeline combining DECODE localization, MAGIK GNN tracking, and ROI-based calcium analysis. The project is more mature than `piezo1_calcium_tracker` with more scripts (16+) including data download utilities, debug scripts, and evaluation tools. Key modules: `decode_net.py`, `magik_gnn.py` (plus `magik_gnn_old.py`), `calcium_roi.py`, `synthetic_generator.py`, `decode_dataset.py`, `magik_dataset.py`. Has a `scripts/` directory with training, evaluation, resume, and pipeline scripts. Leftover files (`magik_gnn_old.py`, `renumberDataFiles.py`) suggest active development with some cleanup needed.

## Short-term Improvements
- [x] Remove or archive `magik_gnn_old.py` -- keep only the current `magik_gnn.py`
- [x] Clean up `scripts/renumberDataFiles.py` and `scripts/10_fix_corrupted_data.py` -- move to a `scripts/utils/` subfolder
- [x] Add docstrings to `decode_dataset.py` and `magik_dataset.py` data loading functions
- [x] Add input validation in `calcium_roi.py` for ROI extraction (bounds checking, NaN handling)
- [x] Create `tests/` directory with forward-pass shape tests for DECODE and MAGIK networks
- [ ] Add `requirements.txt` with torch-geometric and torch-scatter version pins (already existed with pins)
- [x] Create `INTERFACE.md` for project navigation

## Feature Enhancements
- [ ] Implement `signal_detection.py` (listed in README but missing) for calcium peak detection with configurable thresholds
- [ ] Add `track_analysis.py` (listed in README but missing) for MSD, diffusion coefficients, and confinement detection
- [ ] Add wandb/TensorBoard integration to `02_train_decode.py` and `03_train_magik.py`
- [ ] Implement `06_visualize_results.py` with track overlay on TIFF stacks and calcium trace plots
- [ ] Add confidence calibration for DECODE detection probabilities
- [ ] Implement batch inference in `04_run_pipeline.py` for processing multiple FOVs
- [ ] Add support for variable ROI sizes in `calcium_roi.py` based on PSF width

## Long-term Vision
- [ ] Build a unified CLI with subcommands: `piezo1-magik train-decode`, `piezo1-magik track`, etc.
- [ ] Add napari plugin for interactive track inspection and calcium event annotation
- [ ] Implement online/streaming MAGIK tracking for real-time acquisition
- [ ] Add multi-channel support beyond dual-channel (e.g., FRET, multi-color labeling)
- [ ] Publish as pip-installable package with pretrained model weights
- [ ] Benchmark against ISBI Particle Tracking Challenge datasets (scripts 11-14 download them)

## Technical Debt
- [ ] Consolidate `02_train_decode.py` and `02b_resume_decode_training.py` into a single script with `--resume` flag
- [ ] Same for `03_train_magik.py` / `03b_resume_magik_training.py`
- [x] `scripts/debug_decode.py` should be moved to `tests/` or removed
- [ ] Missing `__init__.py` in `piezo1_magik/tracking/` (listed in README structure but directory may not exist)
- [ ] Standardize config loading -- some scripts use YAML, others have hardcoded parameters
- [x] Add `pyproject.toml` for modern packaging alongside `setup.py`
