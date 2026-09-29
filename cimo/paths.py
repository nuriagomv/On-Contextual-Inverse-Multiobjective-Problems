"""Locations that should not depend on the process working directory."""

from pathlib import Path

# cimo/paths.py -> repository root is the parent of the package directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "cache"
RESULTS_DIR = REPO_ROOT / "results"
