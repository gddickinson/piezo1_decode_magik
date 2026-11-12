// Node test: psf_engine.js reproduces Gaussian2DPSF and Airy2DPSF from piezo1_magik/utils/psf_models.py
// (tests/fixtures/engine_cases.json). Run: node psf_engine.test.mjs
import { airyPsf, gaussianPsf, gaussianSigmaPx, PSF_DEFAULTS } from './psf_engine.js';
import { close, fixture } from './test_util.mjs';

let checks = 0;
for (const c of fixture().psf) {
  const o = { ...PSF_DEFAULTS, ...c.opts };
  close(gaussianSigmaPx(o), c.sigma_px, 'sigma_px');
  const g = gaussianPsf(o, c.x, c.y, c.photons, c.H, c.W), a = airyPsf(o, c.x, c.y, c.photons, c.H, c.W);
  c.gauss.flat().forEach((v, i) => { close(g[i], v, `gauss[${i}]`, 1e-9); checks++; });
  // J1 is a rational approximation (|error| ~ 1e-8): compare to 1e-6 of the peak value
  const peak = Math.max(...c.airy.flat());
  c.airy.flat().forEach((v, i) => { close(a[i] / peak, v / peak, `airy[${i}]`, 1e-6); checks++; });
}
console.log(`psf_engine: ${checks} checks ok`);
