"""Compatibility entry point for large SNN experiment batches."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.training.run_big_snn_experiments", run_name="__main__")

