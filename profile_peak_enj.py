"""Estimate peak energy and resource metrics for trained EEG classification models."""

import argparse
import csv
import os
import threading
import time

import torch
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetPowerUsage,
    nvmlInit,
    nvmlShutdown,
)
from torch.utils.data import ConcatDataset, DataLoader

from tools import functional
from src import models
from src.data import EEGset


MODEL_NAMES = [
    "ShallowConvNet",
    "deepconv",
    "EEGNet",
    "CUPY_SNN_LIF_READOUT",
    "CUPY_SNN_PLIF",
]


def build_model(model_name: str, device: torch.device):
    if model_name in ["ShallowConvNet", "deepconv", "EEGNet"]:
        model = getattr(models, model_name)(in_channels=22, time_step=1000, classes_num=4)
    elif model_name == "FBCNet":
        model = getattr(models, model_name)(nChan=22, nTime=1000, nClass=4)
    elif model_name == "CUPY_SNN_LIF_READOUT":
        model = getattr(models, model_name)(in_channels=22, out_num=4, time_step=1000, beta=2)
    elif model_name == "CUPY_SNN_PLIF":
        model = getattr(models, model_name)(in_channels=22, out_num=4, time_step=1000, beta=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


class NvmlPowerMonitor:
    def __init__(self, nvml_handle, interval_s: float = 0.05):
        self.nvml_handle = nvml_handle
        self.interval_s = interval_s
        self.samples = []
        self._stop = False

    def _run(self):
        while not self._stop:
            try:
                power_w = nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
                self.samples.append((time.perf_counter(), power_w))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self):
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop = True
        self._thread.join(timeout=2.0)

    @property
    def peak_power_w(self):
        if not self.samples:
            return 0.0
        return max(power for _, power in self.samples)

    @property
    def mean_power_w(self):
        if not self.samples:
            return 0.0
        return sum(power for _, power in self.samples) / len(self.samples)


def build_test_loader(root_path: str, model_name: str, batch_size: int = 128):
    if model_name == "FBCNet":
        base = os.path.join(root_path, "filterbank")
    else:
        base = root_path

    datasets = [
        EEGset(
            root_path=base + "/",
            pick_id=(sid,),
            settup="test",
            T=1000,
            loo=False,
            EA=False,
            all_id=range(1, 10),
        )
        for sid in range(1, 10)
    ]
    merged = ConcatDataset(datasets)
    return DataLoader(merged, batch_size=batch_size, shuffle=False, drop_last=False)


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def profile_model(model_name: str, loader: DataLoader, device: torch.device, nvml_handle):
    model = build_model(model_name, device)
    model.eval()

    total_params, trainable_params = count_parameters(model)

    with torch.no_grad():
        warmup_batches = 3
        for idx, (x, _) in enumerate(loader):
            if idx >= warmup_batches:
                break
            x = x.to(device)
            _ = model(x)
            torch.cuda.synchronize(device)
            functional.reset_net(model)

        total_time_s = 0.0
        total_energy_j = 0.0
        total_samples = 0

        with NvmlPowerMonitor(nvml_handle, interval_s=0.05) as monitor:
            for x, _ in loader:
                x = x.to(device)

                torch.cuda.synchronize(device)
                p0_w = nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
                t0 = time.perf_counter()

                _ = model(x)

                torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                p1_w = nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0

                dt = t1 - t0
                total_time_s += dt
                total_energy_j += 0.5 * (p0_w + p1_w) * dt
                total_samples += x.shape[0]
                functional.reset_net(model)

    avg_power_w = total_energy_j / total_time_s if total_time_s > 0 else 0.0
    throughput_sps = total_samples / total_time_s if total_time_s > 0 else 0.0
    energy_per_sample_mj = (total_energy_j / total_samples * 1000.0) if total_samples > 0 else 0.0

    return {
        "model": model_name,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "samples": total_samples,
        "inference_time_s": total_time_s,
        "throughput_samples_per_s": throughput_sps,
        "energy_j": total_energy_j,
        "avg_power_w": avg_power_w,
        "peak_power_w": monitor.peak_power_w,
        "mean_power_w": monitor.mean_power_w,
        "power_samples": len(monitor.samples),
        "energy_per_sample_mj": energy_per_sample_mj,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile inference energy/power.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(repo_root, "data", "BNCI2014001", "250Hz_preprocess_eeg")
    if not os.path.isdir(data_root):
        raise RuntimeError(f"Missing data directory: {data_root}")

    device = torch.device(f"cuda:{args.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiling run.")

    nvmlInit()
    nvml_handle = nvmlDeviceGetHandleByIndex(args.device)

    results = []
    try:
        for model_name in MODEL_NAMES:
            loader = build_test_loader(data_root, model_name, batch_size=args.batch_size)
            result = profile_model(model_name, loader, device, nvml_handle)
            print(result)
            results.append(result)
    finally:
        nvmlShutdown()

    csv_path = os.path.join(repo_root, "bnci2014001_inference_profile.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "params_total",
                "params_trainable",
                "samples",
                "inference_time_s",
                "throughput_samples_per_s",
                "energy_j",
                "avg_power_w",
                "peak_power_w",
                "mean_power_w",
                "power_samples",
                "energy_per_sample_mj",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
