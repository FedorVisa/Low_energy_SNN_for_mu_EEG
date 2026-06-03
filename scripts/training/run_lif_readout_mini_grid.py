"""Compatibility entry point for compact LIF readout grid experiments."""

import sys
from pathlib import Path
from runpy import run_module

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    run_module("src.training.run_lif_readout_mini_grid", run_name="__main__")

