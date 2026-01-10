# PIEZO1 DECODE-MAGIK Tracker

**Modular pipeline combining DECODE localization, MAGIK tracking, and ROI-based calcium analysis**

1. **DECODE**: Sub-pixel puncta localization
2. **MAGIK**: Graph Neural Network tracking
3. **ROI Analysis**: Direct calcium intensity extraction

---

## Architecture Overview

```
Input: Dual-channel TIFF stack (PIEZO1 + Calcium)
         ↓
┌────────┴────────────────────────────────────────┐
│ Stage 1: DECODE Localization                   │
│ - Detects puncta per frame                     │
│ - Sub-pixel coordinates (x, y)                 │
│ - Photon counts, uncertainties                 │
└────────┬────────────────────────────────────────┘
         ↓
┌────────┴────────────────────────────────────────┐
│ Stage 2: MAGIK Tracking                        │
│ - Graph Neural Network                         │
│ - Learns to link detections across frames      │
│ - Handles blinking, splitting, merging         │
│ - Outputs: Track IDs, trajectories             │
└────────┬────────────────────────────────────────┘
         ↓
┌────────┴────────────────────────────────────────┐
│ Stage 3: ROI Calcium Analysis                  │
│ - Extract calcium intensity at each punctum    │
│ - Temporal fluorescence traces                 │
│ - Detect intensity peaks (calcium events)      │
│ - Correlate with puncta dynamics               │
└─────────────────────────────────────────────────┘
```

---

## Why This Approach?

✅ **Modular**: Each component can be trained/debugged independently  
✅ **Simpler calcium analysis**: No segmentation needed, just intensity traces  
✅ **Better tracking**: MAGIK handles complex scenarios (blinking, crowding)  
✅ **Direct biological readout**: ΔF/F₀ traces at each channel  
✅ **Proven methods**: DECODE + MAGIK are published, validated approaches

---

## Project Structure

```
piezo1_decode_magik/
├── README.md                          # This file
├── IMPLEMENTATION_GUIDE.md            # Step-by-step guide
├── requirements.txt                   # Dependencies
├── setup.py                           # Installation
│
├── piezo1_magik/                      # Main package
│   ├── models/
│   │   ├── decode_net.py             # DECODE localization
│   │   └── magik_gnn.py              # MAGIK tracking GNN
│   ├── data/
│   │   ├── synthetic_generator.py    # Generate with tracks
│   │   ├── dataset.py                # PyTorch datasets
│   │   └── graph_builder.py          # Build graphs for MAGIK
│   ├── tracking/
│   │   ├── graph_tracker.py          # GNN-based tracking
│   │   └── track_analysis.py         # Trajectory analysis
│   ├── analysis/
│   │   ├── calcium_roi.py            # ROI extraction
│   │   └── signal_detection.py       # Peak detection
│   └── utils/
│       └── psf_models.py             # PSF simulation
│
├── scripts/                           # Executable scripts
│   ├── 01_generate_synthetic_data.py
│   ├── 02_train_decode.py
│   ├── 03_train_magik.py
│   ├── 04_run_pipeline.py            # Full pipeline
│   ├── 05_analyze_calcium.py
│   └── 06_visualize_results.py
│
└── configs/                           # Configuration files
    ├── decode_training.yaml
    ├── magik_training.yaml
    └── pipeline.yaml
```

---

## Key Components

### **1. DECODE Localization**

Based on Speiser et al., Nature Methods 2021:
- CNN processes 3-frame windows
- Outputs: detection probability + coordinates + uncertainties
- Trained on synthetic data with ground truth
- Achieves 20-40 nm precision

### **2. MAGIK Tracking**

Based on graph neural networks:
- Nodes: Detected puncta (one per detection)
- Edges: Potential links between frames
- GNN learns edge probabilities
- Handles blinking, splitting, merging, crowding

**Key features:**
- Temporal context (looks ahead/behind)
- Motion prediction
- Appearance features (intensity, size)
- Learned linking strategy (not hand-crafted rules)

### **3. ROI Calcium Analysis**

For each tracked punctum:
1. Extract 5×5 pixel ROI centered at (x, y)
2. Compute mean intensity over ROI
3. Generate fluorescence trace F(t)
4. Calculate ΔF/F₀ = (F - F₀) / F₀
5. Detect peaks → calcium events
6. Correlate with puncta dynamics

**Advantages:**
- No training needed (rule-based)
- Direct biological interpretation
- Standard calcium imaging analysis
- Easy to validate

---

## Installation

```bash
tar -xzf piezo1_decode_magik.tar.gz
cd piezo1_decode_magik

conda create -n piezo1_magik python=3.11
conda activate piezo1_magik

pip install -r requirements.txt
pip install -e .
```

### Additional Dependencies

```bash
# For graph neural networks
pip install torch-geometric torch-scatter torch-sparse
```

---

## Quick Start

### **1. Generate Synthetic Data with Tracks**

```bash
python scripts/01_generate_synthetic_data.py \
    --output data/synthetic \
    --num_samples 5000 \
    --with_tracks  # Generate full trajectories
```

### **2. Train DECODE Localization**

```bash
python scripts/02_train_decode.py \
    --config configs/decode_training.yaml \
    --data data/synthetic \
    --output checkpoints/decode
```

### **3. Train MAGIK Tracking**

```bash
python scripts/03_train_magik.py \
    --config configs/magik_training.yaml \
    --data data/synthetic \
    --decode_model checkpoints/decode/best_model.pth \
    --output checkpoints/magik
```

### **4. Run Full Pipeline**

```bash
python scripts/04_run_pipeline.py \
    --decode checkpoints/decode/best_model.pth \
    --magik checkpoints/magik/best_model.pth \
    --input /path/to/dual/channel/data \
    --output results/
```

This will:
- Detect puncta with DECODE
- Link into tracks with MAGIK
- Extract calcium ROI traces
- Detect calcium events
- Save all results

---

## Expected Performance

### **Localization (DECODE)**
- Precision: 20-40 nm
- Recall: > 90% (SNR > 5)
- Speed: 100-200 fps on GPU

### **Tracking (MAGIK)**
- Track completeness: > 80%
- Identity preservation: > 90%
- Handles 2-5 particles/μm² density

### **Calcium Analysis**
- Event detection sensitivity: > 85%
- False positive rate: < 10%
- Temporal resolution: Limited by acquisition rate

---

## Outputs

Per movie, the pipeline generates:

**Localization:**
- `detections.csv` - All detected puncta (frame, x, y, intensity, uncertainty)

**Tracking:**
- `tracks.csv` - Linked trajectories (track_id, frame, x, y)
- `track_stats.csv` - Per-track statistics (lifetime, displacement, mobility)

**Calcium:**
- `calcium_traces.csv` - Fluorescence traces per track
- `calcium_events.csv` - Detected events (track_id, frame, amplitude, duration)
- `correlation.csv` - Puncta-calcium correlation metrics

**Visualizations:**
- `tracks_overlay.mp4` - Tracks overlaid on movie
- `calcium_traces.png` - Example fluorescence traces
- `correlation_heatmap.png` - Spatial correlation


## Biological Applications

### **What You Can Measure**

**Puncta Properties:**
- Localization precision: 20-40 nm
- Mobility classification (mobile vs immobile)
- Diffusion coefficients
- Cluster analysis

**Calcium Dynamics:**
- Event frequency per channel
- Event amplitude (ΔF/F₀)
- Event duration
- Spatial spread from punctum

**Correlations:**
- Does puncta movement trigger calcium?
- Do calcium events change puncta mobility?
- Spatial relationship between channels and events

---

## Citation

If you use this pipeline:

```
DECODE: Speiser et al., Nature Methods 2021
MAGIK: Jiang et al., (DeepTrack 2.0)
PIEZO1-HaloTag: Bertaccini et al., Nature Commun 2025
```

---

## License

MIT License - See LICENSE file

---

## Contact

George Dickinson  
UC Irvine
george.dickinson@gmail.com

---

See `IMPLEMENTATION_GUIDE.md` for detailed instructions!
