"""Compatibility entry point for the official CuPy PLIF trainer."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.training.run_official_plif", run_name="__main__")

