"""
Point Spread Function Models for Synthetic Data Generation

Provides Gaussian and Airy PSF models for simulating realistic puncta.
"""

import numpy as np
from scipy.special import j1


class Gaussian2DPSF:
    """2D Gaussian PSF model (fast approximation)."""
    
    def __init__(self, wavelength_nm=646, NA=1.49, pixel_size_nm=130):
        """
        Args:
            wavelength_nm: Emission wavelength (nm)
            NA: Numerical aperture
            pixel_size_nm: Pixel size (nm)
        """
        # Abbe diffraction limit
        fwhm_nm = 0.61 * wavelength_nm / NA
        
        # Convert to sigma (pixels)
        self.sigma_px = (fwhm_nm / pixel_size_nm) / 2.355
        self.pixel_size_nm = pixel_size_nm
    
    def generate(self, x, y, photons=1000, image_size=(64, 64)):
        """
        Generate PSF at position (x, y).
        
        Args:
            x, y: Position in pixels (can be sub-pixel)
            photons: Number of photons
            image_size: Image size (H, W)
            
        Returns:
            psf: (H, W) PSF image
        """
        H, W = image_size
        
        # Create coordinate grids
        yy, xx = np.ogrid[:H, :W]
        
        # Gaussian PSF
        r2 = (xx - x)**2 + (yy - y)**2
        psf = np.exp(-r2 / (2 * self.sigma_px**2))
        
        # Normalize to photon count
        psf = psf / psf.sum() * photons
        
        return psf


class Airy2DPSF:
    """2D Airy disk PSF model (more accurate)."""
    
    def __init__(self, wavelength_nm=646, NA=1.49, pixel_size_nm=130):
        """
        Args:
            wavelength_nm: Emission wavelength (nm)
            NA: Numerical aperture
            pixel_size_nm: Pixel size (nm)
        """
        self.wavelength_nm = wavelength_nm
        self.NA = NA
        self.pixel_size_nm = pixel_size_nm
        
        # First zero of Airy disk
        self.airy_radius_nm = 0.61 * wavelength_nm / NA
        self.airy_radius_px = self.airy_radius_nm / pixel_size_nm
    
    def generate(self, x, y, photons=1000, image_size=(64, 64)):
        """Generate Airy disk PSF."""
        H, W = image_size
        
        yy, xx = np.ogrid[:H, :W]
        
        # Distance from center
        r = np.sqrt((xx - x)**2 + (yy - y)**2)
        
        # Convert to dimensionless units
        v = 2 * np.pi * self.NA * r * self.pixel_size_nm / self.wavelength_nm
        
        # Airy function: (2*J1(v)/v)^2
        # Handle v=0 case
        psf = np.zeros_like(v)
        mask = v > 1e-10
        psf[mask] = (2 * j1(v[mask]) / v[mask])**2
        psf[~mask] = 1.0  # Limit as v->0
        
        # Normalize
        psf = psf / psf.sum() * photons
        
        return psf


def add_noise(image, baseline=100, read_noise=5):
    """
    Add realistic camera noise.
    
    Args:
        image: Clean image (photons)
        baseline: Camera baseline (ADU)
        read_noise: Read noise std (electrons)
        
    Returns:
        noisy: Noisy image (ADU)
    """
    # Poisson noise (shot noise)
    noisy = np.random.poisson(image)
    
    # Gaussian read noise
    noisy = noisy + np.random.randn(*image.shape) * read_noise
    
    # Add baseline
    noisy = noisy + baseline
    
    # Convert to uint16
    noisy = np.clip(noisy, 0, 65535).astype(np.uint16)
    
    return noisy


# Test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    print("Testing PSF models...")
    
    # Gaussian PSF
    gauss_psf = Gaussian2DPSF()
    psf_gauss = gauss_psf.generate(32.5, 32.5, photons=1000)
    
    # Airy PSF
    airy_psf = Airy2DPSF()
    psf_airy = airy_psf.generate(32.5, 32.5, photons=1000)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(psf_gauss, cmap='hot')
    axes[0].set_title('Gaussian PSF')
    axes[0].axis('off')
    
    axes[1].imshow(psf_airy, cmap='hot')
    axes[1].set_title('Airy PSF')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/psf_comparison.png', dpi=150)
    print("✅ PSF models tested, saved to /tmp/psf_comparison.png")
    
    # Test noise
    noisy = add_noise(psf_gauss)
    print(f"✅ Noise added: range {noisy.min()} - {noisy.max()}")
