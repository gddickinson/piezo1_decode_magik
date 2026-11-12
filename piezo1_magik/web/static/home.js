// Home: hero, real counts (scripts, parameters, demo data) and cards to the pages.
import { api, app, isStatic, route } from './core.js';

route('home', async () => {
  const i = await api('/info');
  const d = i.demo;
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">PIEZO1 puncta · deep-learning localization · graph tracking · calcium ROIs</p>
    <h1 class="glow">PIEZO1 DECODE-MAGIK</h1>
    <p class="lead">A modular pipeline for dual-channel microscopy movies (PIEZO1-HaloTag + a calcium indicator):
      <b>DECODE</b> finds puncta with sub-pixel precision, <b>MAGIK</b>, a graph neural network, links them into tracks,
      and an <b>ROI analysis</b> reads the calcium signal at every tracked punctum and detects calcium events.</p>
    <a class="btn" href="#/pipeline">See the pipeline</a><a class="btn alt" href="#/roi">Try the ROI analysis</a><a class="btn alt" href="#/params">All parameters</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${i.stages}</b><span>pipeline stages</span></div>
    <div class="tile"><b>${i.n_scripts}</b><span>numbered scripts (data, training, evaluation, pipeline)</span></div>
    <div class="tile"><b>${i.n_config_values}</b><span>values in configs/*.yaml</span></div>
    <div class="tile"><b>${i.n_options}</b><span>command-line options</span></div>
    <div class="tile"><b>${i.facts.pixel_size_nm} nm</b><span>pixel size assumed by the PSF models</span></div>
  </div>
  <div class="callout">${isStatic()
    ? 'This static site runs the lightweight stages in your browser: the calcium ROI analysis, the track linking and gap filling steps around MAGIK, and the PSF models, all JS ports checked against the Python code. DECODE and MAGIK are deep-learning tools with their own licences and need PyTorch and trained checkpoints, so they run only in the local Python install.'
    : 'Local app: the ROI and track analyses on these pages run in the repo\'s Python code. DECODE and MAGIK themselves run through the scripts with your trained checkpoints (see Pipeline).'}</div>
  <h2>Pages</h2>
  <div class="grid">
    <a class="card" href="#/pipeline"><b>Pipeline</b><span>The three stages, what each takes and produces, the networks and the linking rules, with the source files.</span></a>
    <a class="card" href="#/params"><b>Parameters</b><span>Every value in configs/*.yaml and every script option, with units (px, frames, nm, photons) and px in µm.</span></a>
    <a class="card" href="#/roi"><b>ROI calcium analysis</b><span>Trace, ΔF/F₀ and event detection at each tracked punctum of a synthetic demo movie; change the ROI size, baseline and thresholds.</span></a>
    <a class="card" href="#/tracks"><b>Track linking and gap filling</b><span>Candidate graph, greedy linking and gap filling on the demo's detections, scored against the ground truth.</span></a>
    <a class="card" href="#/psf"><b>PSF models</b><span>The Gaussian and Airy PSFs used to make the synthetic training data, from wavelength, NA and pixel size.</span></a>
    <a class="card" href="#/about"><b>About</b><span>How to run the scripts, the desktop pipeline and this web app; citations and licence.</span></a>
  </div>
  <h2>Demo data</h2>
  <p class="demo-label">Synthetic demo, not a measurement: ${d.frames} frames of ${d.width} × ${d.height} px, ${d.tracks} PIEZO1 tracks
    (${d.detections} detections, with blinking) from the repo's generator, and a calcium channel with ${d.events} events placed on tracked puncta.</p>`;
});
