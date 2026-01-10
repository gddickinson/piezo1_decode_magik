"""
Setup script for PIEZO1 DECODE-MAGIK Tracker
"""

from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="piezo1-decode-magik",
    version="0.1.0",
    author="George Dickinson",
    author_email="your.email@uci.edu",
    description="DECODE localization + MAGIK tracking + ROI calcium analysis for PIEZO1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "tifffile>=2023.0.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
