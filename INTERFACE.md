# PIEZO1 DECODE-MAGIK -- Interface Map

## Project Overview
Modular pipeline combining DECODE localization, MAGIK GNN tracking, and ROI-based
calcium signal analysis for PIEZO1 puncta in dual-channel microscopy data.

## Package: `piezo1_magik/`

### `models/`
- **`decode_net.py`** -- DECODE U-Net for puncta localization
  - `FrameAnalysisNet` -- first U-Net, per-frame spatial features
  - `TemporalContextNet` -- second U-Net, 3-frame temporal integration
  - `DECODENet` -- full network with output heads (prob, offset, photons, uncertainty)
- **`magik_gnn.py`** -- Graph neural network for particle tracking
  - `EdgeConvLayer` -- message-passing layer (no torch_geometric dependency)
  - `MAGIKNet` -- full GNN with node/edge encoders and edge classifier
  - `build_tracking_graph()` -- builds temporal graph from detections

### `data/`
- **`decode_dataset.py`** -- PyTorch Dataset for DECODE training
  - `DECODEDataset` -- loads synthetic 3-frame windows with ground truth
  - `create_dataloaders()` -- creates train/val DataLoaders
- **`magik_dataset.py`** -- PyTorch Dataset for MAGIK training
  - `MAGIKDataset` -- runs DECODE on movies, builds graphs with GT labels
  - `collate_graphs()` -- batches multiple graphs via disjoint union
- **`synthetic_generator.py`** -- generates synthetic microscopy data

### `analysis/`
- **`calcium_roi.py`** -- ROI-based calcium signal extraction and analysis
  - `CalciumROIAnalyzer` -- extracts traces, computes dF/F0, detects events
  - `analyze_all_tracks()` -- batch analysis for all tracks

### `utils/`
- **`psf_models.py`** -- point spread function models

## Scripts: `scripts/`
- `01_generate_synthetic_data.py` -- synthetic data generation
- `01b_check_training_data.py` -- data validation
- `02_train_decode.py` / `02b_resume_decode_training.py` -- DECODE training
- `02c_evaluate_decode.py` -- DECODE evaluation
- `03_train_magik.py` / `03b_resume_magik_training.py` -- MAGIK training
- `03c_evaluate_magik.py` -- MAGIK evaluation
- `04_run_pipeline.py` -- full pipeline (detect + track)
- `06_visualize_results.py` -- result visualization
- `07_analyze_tracks.py` -- track analysis (MSD, diffusion)
- `08_fill_track_gaps.py` -- gap filling in tracks
- `09_run_complete_pipeline.py` -- end-to-end pipeline
- `11-14_download_*.py` -- public dataset downloaders

### `scripts/utils/`
- `renumber_data_files.py` -- renumber synthetic data folders
- `fix_corrupted_data.py` -- find/remove corrupted TIFF files

## Tests: `tests/`
- `test_decode_net.py` -- forward-pass shape tests for DECODE
- `test_magik_gnn.py` -- forward-pass shape tests for MAGIK
- `debug_decode.py` -- interactive model debugging (moved from scripts/)

## Config: `configs/`
YAML configuration files for training runs.

## Archived: `_archive/`
- `magik_gnn_old.py` -- previous version of MAGIK GNN
