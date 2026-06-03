"""Compatibility entry point for lab-made MI dataset SNN training."""

import sys
from pathlib import Path
from runpy import run_module

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    run_module("src.training.train_lab_made_snn", run_name="__main__")

