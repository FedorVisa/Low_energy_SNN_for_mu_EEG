"""Compatibility entry point for lab-made MI dataset SNN training."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.training.train_lab_made_snn", run_name="__main__")

