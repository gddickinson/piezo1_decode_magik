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
