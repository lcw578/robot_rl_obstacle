"""Paths shared by the simulation and its command-line entry points."""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / 'config'
DEFAULT_CONFIG = CONFIG_DIR / 'paper_simulation.yaml'


def repo_relative(path):
    """Represent a file path relative to the checkout in saved metadata."""
    return Path(os.path.relpath(Path(path).resolve(), PROJECT_ROOT)).as_posix()
