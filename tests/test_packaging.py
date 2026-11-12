"""Packaging regression tests: `pip install -e .` must find the piezo1_magik package and a valid build backend."""
import importlib
import tomllib
from pathlib import Path

from setuptools import find_packages

ROOT = Path(__file__).resolve().parent.parent


def test_find_packages_includes_piezo1_magik():
    pkgs = set(find_packages(where=str(ROOT), exclude=["tests", "tests.*"]))
    for name in ("piezo1_magik", "piezo1_magik.analysis", "piezo1_magik.data", "piezo1_magik.models",
                 "piezo1_magik.utils"):
        assert name in pkgs, f"{name} not found by find_packages: {sorted(pkgs)}"


def test_pyproject_build_backend_importable():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    module, _, attr = cfg["build-system"]["build-backend"].partition(":")
    mod = importlib.import_module(module)
    if attr:
        getattr(mod, attr)
