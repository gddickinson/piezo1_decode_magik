"""PSF width tests: the Gaussian approximation must have the FWHM of the Airy disk it approximates.

The Airy intensity (2 J1(v)/v)^2 with v = 2 pi NA r / lambda falls to half its peak at v = 1.6163, so its FWHM is
2 * 1.6163 / (2 pi) * lambda / NA = 0.5145 lambda / NA (Born & Wolf, Principles of Optics, sec. 8.5.2;
the usual 0.51 lambda/NA). 0.61 lambda/NA is the radius of the first dark ring (Rayleigh criterion), not the FWHM.
"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import j1

from piezo1_magik.utils.psf_models import Airy2DPSF, Gaussian2DPSF


def airy_fwhm_nm(wavelength_nm, NA):
    v_half = brentq(lambda v: (2 * j1(v) / v) ** 2 - 0.5, 0.5, 3.0)
    return 2 * v_half * wavelength_nm / (2 * np.pi * NA)


@pytest.mark.parametrize("wl,na,px", [(646, 1.49, 130), (520, 1.2, 108), (700, 0.9, 65)])
def test_gaussian_fwhm_matches_airy_fwhm(wl, na, px):
    g = Gaussian2DPSF(wavelength_nm=wl, NA=na, pixel_size_nm=px)
    fwhm_nm = g.sigma_px * px * 2 * np.sqrt(2 * np.log(2))
    assert fwhm_nm == pytest.approx(airy_fwhm_nm(wl, na), rel=1e-3)
    assert g.fwhm_nm == pytest.approx(fwhm_nm, rel=1e-12)


def test_airy_model_half_maximum_at_that_radius():
    """The repo's own Airy2DPSF profile drops to half its peak at FWHM/2 (sampled finely)."""
    wl, na, px = 646, 1.49, 10.0  # 10 nm pixels for fine sampling
    a = Airy2DPSF(wavelength_nm=wl, NA=na, pixel_size_nm=px)
    img = a.generate(100, 100, photons=1.0, image_size=(201, 201))
    row = img[100] / img[100, 100]
    r_half = np.interp(0.5, row[100:][::-1], np.arange(101)[::-1]) * px  # monotone inside the first ring
    assert 2 * r_half == pytest.approx(airy_fwhm_nm(wl, na), rel=1e-2)
    assert a.airy_radius_nm == pytest.approx(0.61 * wl / na)  # first dark ring stays 0.61 lambda/NA
