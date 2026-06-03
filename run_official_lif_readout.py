"""Compatibility entry point for the official CuPy LIF readout trainer."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.training.run_official_lif_readout", run_name="__main__")

