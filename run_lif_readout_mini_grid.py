"""Compatibility entry point for compact LIF readout grid experiments."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.training.run_lif_readout_mini_grid", run_name="__main__")

