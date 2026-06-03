"""Compatibility entry point for large SNN experiment batches."""

import sys
from pathlib import Path
from runpy import run_module

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    run_module("src.training.run_big_snn_experiments", run_name="__main__")

