// PSF models: a JS port of piezo1_magik/utils/psf_models.py (Gaussian2DPSF, Airy2DPSF). Tested against Python by
// psf_engine.test.mjs. Positions in pixels (pixel centres), wavelength and pixel size in nm.

export const PSF_DEFAULTS = { wavelength_nm: 646, NA: 1.49, pixel_size_nm: 130 };

// Gaussian2DPSF: the code takes 0.61 * lambda / NA as the FWHM and sigma = FWHM / 2.355 (in pixels)
export function gaussianSigmaPx({ wavelength_nm, NA, pixel_size_nm }) {
  const fwhmNm = 0.61 * wavelength_nm / NA;
  return (fwhmNm / pixel_size_nm) / 2.355;
}

// Airy2DPSF: radius of the first zero, 0.61 * lambda / NA
export const airyRadiusNm = ({ wavelength_nm, NA }) => 0.61 * wavelength_nm / NA;

// Bessel J1 (Numerical Recipes rational approximations, |error| ~ 1e-8)
export function besselJ1(x) {
  const ax = Math.abs(x);
  if (ax < 8.0) {
    const y = x * x;
    const a = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
    const b = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
    return a / b;
  }
  const z = 8.0 / ax, y = z * z, xx = ax - 2.356194491;
  const a = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
  const b = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
  const r = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * a - z * Math.sin(xx) * b);
  return x < 0 ? -r : r;
}

function normalise(img, photons) {
  const s = img.reduce((a, b) => a + b, 0);
  return img.map(v => (v / s) * photons);
}

// generate(x, y, photons, image_size=(H, W)) -> Float64Array H*W, row-major
export function gaussianPsf(opts, x, y, photons, H, W) {
  const s = gaussianSigmaPx(opts), img = new Float64Array(H * W);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) img[r * W + c] = Math.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * s * s));
  return normalise(img, photons);
}

export function airyPsf({ wavelength_nm, NA, pixel_size_nm }, x, y, photons, H, W) {
  const img = new Float64Array(H * W);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const v = 2 * Math.PI * NA * Math.sqrt((c - x) ** 2 + (r - y) ** 2) * pixel_size_nm / wavelength_nm;
    img[r * W + c] = v > 1e-10 ? (2 * besselJ1(v) / v) ** 2 : 1.0;
  }
  return normalise(img, photons);
}
