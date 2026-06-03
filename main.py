"""Compatibility entry point for the main MI EEG training script."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.training.main", run_name="__main__")

