"""Compatibility entry point for full subject and variant grid experiments."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.training.run_full_grid", run_name="__main__")

