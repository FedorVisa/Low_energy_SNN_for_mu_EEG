"""Compatibility entry point for the Norse latency-coded SNN trainer."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.training.run_norse_latency_snn", run_name="__main__")

