"""Compatibility entry point for subject-specific LIF readout tuning."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.training.run_subject_tuning_lif_readout", run_name="__main__")

