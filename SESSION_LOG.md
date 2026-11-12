# Session log

## 2026-11-12 -- web interface and static site (branch `web-interface`)
- Added a web interface (`piezo1_magik/web/`, `run_web.py`, port 8209) and a GitHub Pages static site
  (`scripts/build_static.py`, `.github/workflows/pages.yml`). Pages: Home, Pipeline, Parameters, ROI calcium
  analysis, Track linking and gap filling, PSF models, About.
- Demo data: `piezo1_magik/data/demo_dual_channel.py` (synthetic; labelled on every page that uses it).
- JS ports `roi_engine.js`, `track_engine.js`, `psf_engine.js` checked by node tests against
  `tests/fixtures/engine_cases.json` (written by `scripts/make_engine_fixture.py`; pytest checks it is current).
- Refactor: greedy linking / gap filling / track statistics, copied in scripts 03c, 07, 08, 09, moved to
  `piezo1_magik/tracking/` (verified equivalent on random graphs); track ids now numbered in sorted node order.
- Bugs fixed (each with a regression test, separate commits):
  - packaging: invalid build backend in pyproject.toml and no `piezo1_magik/__init__.py`, so `pip install -e .`
    failed / installed nothing;
  - `04_run_pipeline.py` stage 2 crashed on every run (wrong `build_tracking_graph` arguments, nonexistent
    `MAGIKNet.predict_links`);
  - gap filling stopped after 100 merges;
  - `extract_roi` for a centre far outside the image read pixels from the other side of the frame;
  - even `roi_size` crashed inside `extract_roi` (now rejected up front).
- Docs corrected: README structure/outputs/flags (`--with_tracks`, `--decode_model` did not exist), linking does
  not split/merge, performance figures labelled as targets; `diffusion_coeff` docstring (it is a step std, not D).
- Open for George: Gaussian PSF uses 0.61 lambda/NA as FWHM (Rayleigh radius); ROI zero-padding at image edges
  lowers F; event threshold is k x std(dF/F0) from zero, not from the baseline; 04 links by connected components
  while 03c/07/08/09 use greedy one-to-one linking; `configs/pipeline.yaml` is read by no script.
- Mac mini not used.

## 2026-11-12 (later) -- fixes requested after review (George's "fix any other problems")
- Gaussian PSF: FWHM = 0.5145 lambda/NA (Airy FWHM, Born & Wolf sec. 8.5.2) instead of 0.61 lambda/NA (first-ring
  radius). Default sigma 0.864 -> 0.729 px. **Synthetic training data changes**; models trained on old data saw
  puncta ~19% too wide. Test: tests/test_psf_models.py.
- ROI at image edges: out-of-image pixels excluded from the mean (not zero-padded), edge frames flagged
  (`edge_roi` column). Removes false events on tracks reaching the border.
- Event threshold: median + k x 1.4826 MAD of dF/F0 (was k x std from zero). Web demo: 4 hits / 2 false
  (was 4 / 16); one remaining "false" event is a neighbour's event leaking into the ROI. Duplicate events within one
  span merged.
- 04_run_pipeline.py: shared greedy one-to-one linking (was connected components); reads configs/pipeline.yaml
  with CLI overrides (min_photons, calcium options and save flags now take effect; unused keys removed).
- JS ports (roi_engine.js, psf_engine.js) and the fixture updated with each fix; all node tests pass.
- Still open: the event threshold uses whole-trace median/MAD (no sliding window; drift should go through
  baseline_method='rolling'); 04's max_frame_gap 5 vs MAGIK training gap 2 is documented, not changed.

## 2026-11-12 (follow-up) -- noise window for events, MAGIK frame gap
- `04_run_pipeline.py` `max_frame_gap` default 5 -> 2 (configs/pipeline.yaml and `track_with_magik`), equal to MAGIK's
  training gap (`magik_training.yaml` `max_temporal_gap`); the gap is an MAGIK edge feature, so longer edges were out of
  the training range. Still configurable; a test runs 04's `main()` (stubbed models) and checks the config value and a
  `--max_frame_gap` override reach `build_candidate_graph`.
- `detect_peaks(noise_window=N)`: rolling median + 1.4826 MAD over N frames centred on each frame (NaN frames count
  but give no value; < 3 values or MAD 0 -> whole-trace values); 0/None = whole trace (old behaviour). Default 61
  (`DEFAULT_NOISE_WINDOW`), chosen with `scripts/simulate_noise_window.py` (seed 0, 200 traces x 300 frames per
  scenario, events 3-10 noise sigma): F1 with drift 0.482 (whole) / 0.872 (31) / 0.877 (41) / 0.865 (61) / 0.828 (81);
  without drift 0.964 / 0.899 / 0.927 / 0.950 / 0.955. 61 has the best mean over both scenarios (seeds 0, 1, 2).
  In pipeline.yaml (`calcium.noise_window`), 04 `--noise_window`, the web app (form field, rolling threshold line) and
  the JS port (`roi_engine.js` `noiseLevels`/`detectPeaks`, fixture `noise` cases + 5 more ROI parameter sets).
  Web demo events unchanged at the default (4 in placed events, 2 elsewhere).
- Tests: pytest tests 66 passed; CI=1 test_web 14 passed, 1 skipped; node roi 8617, track 70, psf 1240 checks.
