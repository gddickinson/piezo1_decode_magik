// PSF models of the synthetic-data generator (psf_models.py), computed in the browser by psf_engine.js.
import { app, route, slider } from './core.js';
import { lineChart } from './charts.js';
import { airyPsf, airyRadiusNm, gaussianPsf, gaussianSigmaPx, PSF_DEFAULTS } from './psf_engine.js';

const N = 15, K = 14;

function paint(canvas, img, peak, tint) {
  canvas.width = N * K; canvas.height = N * K;
  const g = canvas.getContext('2d');
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const v = Math.sqrt(img[r * N + c] / peak);  // square-root scale shows the Airy rings
    g.fillStyle = `rgb(${tint[0] * v},${tint[1] * v},${tint[2] * v})`; g.fillRect(c * K, r * K, K, K);
  }
}

route('psf', () => {
  const o = { ...PSF_DEFAULTS, x: 7, photons: 1000 };
  app().innerHTML = `
  <h1>PSF models</h1>
  <p class="lead">The synthetic training movies (<code>synthetic_generator.py</code>) draw each punctum with <code>Gaussian2DPSF</code>;
    <code>Airy2DPSF</code> is the alternative in <code>psf_models.py</code>. Both are normalised to the photon count over the image.
    Computed here by a JS port checked against the Python code.</p>
  <div class="side"><div id="ctl" class="panel"></div>
  <div>
    <div class="two"><div><h3>Gaussian2DPSF</h3><div class="imgwrap"><canvas id="pg" aria-label="Gaussian PSF image"></canvas></div></div>
      <div><h3>Airy2DPSF</h3><div class="imgwrap"><canvas id="pa" aria-label="Airy PSF image"></canvas></div></div></div>
    <p class="small mute">${N} × ${N} px, emitter at x = <span id="xv"></span> px, y = 7 px (pixel centres); square-root grey scale.</p>
    <div class="panel"><h3>Profile through the emitter row</h3><div id="prof"></div>
      <div class="legend"><span><span class="sw" style="background:#ff5a7a"></span>Gaussian</span><span><span class="sw" style="background:#5ab4ff"></span>Airy</span></div></div>
    <dl class="kv panel" id="nums"></dl>
  </div></div>
  <div class="callout">The code computes the Gaussian width from <code>fwhm_nm = 0.61 · λ / NA</code> and σ = FWHM / 2.355. 0.61 λ/NA is the
    radius of the Airy disk's first dark ring (the Rayleigh distance), which <code>Airy2DPSF</code> also uses; the FWHM of an Airy disk is about
    0.51 λ/NA. So the synthetic Gaussian puncta are about 20% wider than an Airy disk with the same λ and NA. This page reports what the code does.</div>`;
  const ctl = document.getElementById('ctl');
  const update = () => {
    const g = gaussianPsf(o, o.x, 7, o.photons, N, N), a = airyPsf(o, o.x, 7, o.photons, N, N);
    const peak = Math.max(...g, ...a);
    paint(document.getElementById('pg'), g, peak, [255, 90, 122]);
    paint(document.getElementById('pa'), a, peak, [90, 180, 255]);
    document.getElementById('xv').textContent = o.x.toFixed(2);
    const row = img => Array.from({ length: N }, (_, c) => img[7 * N + c]);
    document.getElementById('prof').innerHTML = lineChart([{ y: row(g), colour: '#ff5a7a' }, { y: row(a), colour: '#5ab4ff' }],
      { xlabel: 'x (px)', ylabel: 'photons per pixel', label: 'PSF profiles', h: 200 });
    const s = gaussianSigmaPx(o);
    document.getElementById('nums').innerHTML = `
      <dt>"FWHM" used (0.61 λ/NA)</dt><dd>${(0.61 * o.wavelength_nm / o.NA).toFixed(1)} nm</dd>
      <dt>Gaussian σ</dt><dd>${s.toFixed(3)} px = ${(s * o.pixel_size_nm).toFixed(1)} nm</dd>
      <dt>Airy first-zero radius</dt><dd>${(airyRadiusNm(o) / o.pixel_size_nm).toFixed(3)} px = ${airyRadiusNm(o).toFixed(1)} nm</dd>
      <dt>Peak pixel</dt><dd>Gaussian ${Math.max(...g).toFixed(1)}, Airy ${Math.max(...a).toFixed(1)} photons</dd>`;
  };
  slider(ctl, { id: 'wl', label: 'Emission wavelength λ', min: 450, max: 750, step: 1, value: o.wavelength_nm, unit: ' nm' }, v => { o.wavelength_nm = v; update(); });
  slider(ctl, { id: 'na', label: 'Numerical aperture NA', min: 0.5, max: 1.7, step: 0.01, value: o.NA, fmt: v => v.toFixed(2) }, v => { o.NA = v; update(); });
  slider(ctl, { id: 'px', label: 'Pixel size (sample plane)', min: 50, max: 200, step: 1, value: o.pixel_size_nm, unit: ' nm' }, v => { o.pixel_size_nm = v; update(); });
  slider(ctl, { id: 'x', label: 'Emitter x', min: 6, max: 8, step: 0.05, value: o.x, unit: ' px', fmt: v => v.toFixed(2) }, v => { o.x = v; update(); });
  slider(ctl, { id: 'ph', label: 'Photons', min: 100, max: 5000, step: 100, value: o.photons }, v => { o.photons = v; update(); });
  ctl.insertAdjacentHTML('beforeend', '<p class="hint small mute">Defaults from the code: λ 646 nm, NA 1.49, 130 nm/px.</p>');
  update();
});
