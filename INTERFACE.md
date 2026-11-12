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
  - `TrackGenerator`, `generate_synthetic_movie()` -- movie + ground-truth tracks (Gaussian PSF, noise, blinking)
- **`demo_dual_channel.py`** -- `make_demo_dual_channel()`: small synthetic PIEZO1 + calcium movie for the web demo
  and the JS-vs-Python tests (calcium events placed on tracked puncta)

### `tracking/` (NumPy only, no PyTorch)
- **`linking.py`** -- the one linking implementation, used by scripts 03c, 04, 07, 08, 09 and the web interface
  - `build_candidate_graph()` -- MAGIK input graph (nodes [frame,x,y,prob,photons], edges [dx,dy,dist,gap])
  - `greedy_linking()` -- one-to-one linking of edge probabilities into tracks
  - `build_track_fragments()`, `merge_track_fragments()` (gap filling), `fragments_to_node_mapping()`
- **`track_stats.py`** -- `analyze_track_lengths()`, `analyze_gt_tracks()`, `analyze_fragmentation()`,
  `length_categories()`

### `pipeline_config.py`
- `load_pipeline_config()`, `resolve()` -- configs/pipeline.yaml for 04_run_pipeline.py, with command-line overrides
  (`OPTION_KEYS` maps every YAML key to an option; unknown/missing keys raise)

### `analysis/`
- **`calcium_roi.py`** -- ROI-based calcium signal extraction and analysis
  - `CalciumROIAnalyzer` -- extracts traces, computes dF/F0, detects events
  - `analyze_all_tracks()` -- batch analysis for all tracks

### `utils/`
- **`psf_models.py`** -- point spread function models (`Gaussian2DPSF`, `Airy2DPSF`, `add_noise`)

### `web/` -- web interface (FastAPI + plain ES modules), port 8209
- **`app.py`** -- routes under /api (`/info`, `/pipeline`, `/params`, `/demo` GET and exported to the static site;
  POST `/roi`, `/tracks`; local-only `/models` needs PyTorch), `STATIC_ENDPOINTS`
- **`content.py`** -- pipeline stage descriptions; **`params.py`** -- configs/*.yaml and script options (ast) with units
- **`demo.py`** -- demo JSON, `roi_on()`/`run_roi()` (CalciumROIAnalyzer), `run_tracks()` (oracle links + linking
  + gap filling), `clean()` for JSON
- **`static/`** -- `index.html`, `core.js` + `style.css` (shared kit), `main.js`, `static_api.js`, `data.js`,
  `charts.js`, pages (`home`, `pipeline`, `params`, `roi`, `tracks`, `psf`, `about`), JS ports
  `roi_engine.js`, `track_engine.js`, `psf_engine.js` with node tests `*.test.mjs` (+ `test_util.mjs`)
- `run_web.py` (repo root) starts it; `scripts/build_static.py` exports the static site; `scripts/screenshots.py`
  smoke-tests routes; `scripts/make_engine_fixture.py` writes `tests/fixtures/engine_cases.json`

## Scripts: `scripts/`
- `01_generate_synthetic_data.py` -- synthetic data generation
- `01b_check_training_data.py` -- data validation
- `02_train_decode.py` / `02b_resume_decode_training.py` -- DECODE training
- `02c_evaluate_decode.py` -- DECODE evaluation
- `03_train_magik.py` / `03b_resume_magik_training.py` -- MAGIK training
- `03c_evaluate_magik.py` -- MAGIK evaluation
- `04_run_pipeline.py` -- full pipeline (detect + track + calcium), settings from configs/pipeline.yaml
- `06_visualize_results.py` -- result visualization
- `07_analyze_tracks.py` -- track quality (lengths, spans, gaps, fragmentation)
- `08_fill_track_gaps.py` -- gap filling in tracks
- `09_run_complete_pipeline.py` -- end-to-end pipeline
- `11-14_download_*.py` -- public dataset downloaders

### `scripts/utils/`
- `renumber_data_files.py` -- renumber synthetic data folders
- `fix_corrupted_data.py` -- find/remove corrupted TIFF files

## Tests: `tests/`
- `test_decode_net.py` -- forward-pass shape tests for DECODE
- `test_magik_gnn.py` -- forward-pass shape tests for MAGIK
- `test_packaging.py` -- find_packages and build backend
- `test_run_pipeline_tracking.py` -- 04_run_pipeline.py stage 2 runs (untrained MAGIK)
- `test_tracking_linking.py` -- linking, gap filling, track statistics
- `test_calcium_roi.py` -- ROI extraction edge cases, trace/dF/F0/events
- `test_pipeline_config.py` -- pipeline.yaml loading, overrides, options reaching the analysis
- `test_psf_models.py` -- Gaussian FWHM equals the Airy FWHM
- `test_web.py` -- web API, static build, fixture currency, node engine tests
- `debug_decode.py` -- interactive model debugging (moved from scripts/)

## Config: `configs/`
YAML configuration: decode_training.yaml and magik_training.yaml (training, via --config), pipeline.yaml (04_run_pipeline.py).

## Archived: `_archive/`
- `magik_gnn_old.py` -- previous version of MAGIK GNN
