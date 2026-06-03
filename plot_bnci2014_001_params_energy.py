"""Profile and plot parameter and energy estimates for BNCI2014-001 models."""

import argparse
import csv
import json
import math
import re
import subprocess
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import torch

from tools import functional
from src.models.ConvNet import ShallowConvNet, deepconv
from src.models.EEGNet import EEGNet
from src.models.SNNs import CUPY_SNN_LIF_READOUT


BNCI2014_001_CHANNELS = 22
BNCI2014_001_CLASSES = 4
DEFAULT_MODELS = ["ShallowConvNet", "deepconv", "EEGNet", "CUPY_SNN_LIF_READOUT"]
RU_PROPOSED = "\u041f\u0440\u0435\u0434\u043b\u043e\u0436\u0435\u043d\u043d\u0430\u044f SNN"
RU_CONV = "\u0421\u0432\u0435\u0440\u0442\u043e\u0447\u043d\u044b\u0435 \u0441\u0435\u0442\u0438"
RU_TRAINABLE_PARAMS = "\u041e\u0431\u0443\u0447\u0430\u0435\u043c\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b, \u043b\u043e\u0433\u0430\u0440\u0438\u0444\u043c\u0438\u0447\u0435\u0441\u043a\u0430\u044f \u0448\u043a\u0430\u043b\u0430"
RU_ACCURACY = "\u0422\u043e\u0447\u043d\u043e\u0441\u0442\u044c \u043d\u0430 BNCI2014_001 (%)"
RU_ACC_PARAMS_TITLE = "\u0422\u043e\u0447\u043d\u043e\u0441\u0442\u044c \u0432 \u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e\u0441\u0442\u0438 \u043e\u0442 \u0447\u0438\u0441\u043b\u0430 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432"
RU_PEAK_POWER = "\u041f\u0438\u043a\u043e\u0432\u0430\u044f \u043c\u043e\u0449\u043d\u043e\u0441\u0442\u044c GPU \u043f\u0440\u0438 \u0438\u043d\u0444\u0435\u0440\u0435\u043d\u0441\u0435 (\u0412\u0442)"
RU_ACC_POWER_TITLE = "\u0422\u043e\u0447\u043d\u043e\u0441\u0442\u044c \u0432 \u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e\u0441\u0442\u0438 \u043e\u0442 \u044d\u043d\u0435\u0440\u0433\u043e\u043f\u043e\u0442\u0440\u0435\u0431\u043b\u0435\u043d\u0438\u044f"
RU_PEAK_HIST_TITLE = "\u0413\u0438\u0441\u0442\u043e\u0433\u0440\u0430\u043c\u043c\u0430 \u043f\u0438\u043a\u043e\u0432\u043e\u0433\u043e \u044d\u043d\u0435\u0440\u0433\u043e\u043f\u043e\u0442\u0440\u0435\u0431\u043b\u0435\u043d\u0438\u044f"
RU_POWER = "\u041c\u043e\u0449\u043d\u043e\u0441\u0442\u044c GPU \u043f\u0440\u0438 \u0438\u043d\u0444\u0435\u0440\u0435\u043d\u0441\u0435 (\u0412\u0442)"
RU_POWER_BY_MODEL = "\u0420\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u043c\u043e\u0449\u043d\u043e\u0441\u0442\u0438 GPU \u043f\u043e \u043c\u043e\u0434\u0435\u043b\u044f\u043c"
RU_POWER_HIST_TITLE = "\u0420\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u0437\u0430\u043c\u0435\u0440\u043e\u0432 \u043c\u043e\u0449\u043d\u043e\u0441\u0442\u0438 GPU"
RU_DENSITY = "\u041f\u043b\u043e\u0442\u043d\u043e\u0441\u0442\u044c"
DISPLAY_NAMES = {
    "ShallowConvNet": "ShallowConvNet",
    "deepconv": "DeepConvNet",
    "EEGNet": "EEGNet",
    "CUPY_SNN_LIF_READOUT": "Предложенная SNN",
}
DISPLAY_NAMES["CUPY_SNN_LIF_READOUT"] = RU_PROPOSED
MOABB_ROWS = {
    "ShallowConvNet": "ShallowConvNet",
    "deepconv": "DeepConvNet",
    "EEGNet": "EEGNet_8_2",
}


def parse_accuracy_cell(cell):
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", cell or "")
    if not match:
        raise ValueError(f"Cannot parse accuracy value: {cell!r}")
    return float(match.group(1))


def read_moabb_accuracy(csv_path):
    wanted = {row_name: model_name for model_name, row_name in MOABB_ROWS.items()}
    accuracies = {}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pipeline = row["Pipeline"].strip().strip('"')
            if pipeline in wanted:
                accuracies[wanted[pipeline]] = parse_accuracy_cell(row["BNCI2014_001"])
    missing = sorted(set(MOABB_ROWS) - set(accuracies))
    if missing:
        raise KeyError(f"Missing MOABB rows for: {missing}")
    return accuracies


def read_proposed_accuracy(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    model = payload["models"]["CUPY_SNN_LIF_READOUT"]
    return float(model["overall_mean_accuracy_percent"])


def build_model(model_name, time_step, beta):
    if model_name == "ShallowConvNet":
        return ShallowConvNet(
            classes_num=BNCI2014_001_CLASSES,
            in_channels=BNCI2014_001_CHANNELS,
            time_step=time_step,
        )
    if model_name == "deepconv":
        return deepconv(
            classes_num=BNCI2014_001_CLASSES,
            in_channels=BNCI2014_001_CHANNELS,
            time_step=time_step,
        )
    if model_name == "EEGNet":
        return EEGNet(
            classes_num=BNCI2014_001_CLASSES,
            in_channels=BNCI2014_001_CHANNELS,
            time_step=time_step,
        )
    if model_name == "CUPY_SNN_LIF_READOUT":
        return CUPY_SNN_LIF_READOUT(
            in_channels=BNCI2014_001_CHANNELS,
            out_num=BNCI2014_001_CLASSES,
            time_step=time_step,
            beta=beta,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def query_power_w(device):
    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    return float(output.strip().splitlines()[0])


class PowerMonitor:
    def __init__(self, device, interval_s):
        self.device = device
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.perf_counter(), query_power_w(self.device)))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def peak_power_w(self):
        if not self.samples:
            return math.nan
        return max(power for _, power in self.samples)

    @property
    def mean_power_w(self):
        if not self.samples:
            return math.nan
        return sum(power for _, power in self.samples) / len(self.samples)


def reset_if_needed(model):
    try:
        functional.reset_net(model)
    except Exception:
        pass


def benchmark_power(model, batch, warmup, repeat, device, sample_interval_s, min_measure_seconds):
    model.cuda(device).eval()
    batch = batch.cuda(device, non_blocking=True)

    time.sleep(0.75)
    idle_samples = []
    for _ in range(5):
        idle_samples.append(query_power_w(device))
        time.sleep(sample_interval_s)
    idle_power_w = sum(idle_samples) / len(idle_samples)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(batch)
            torch.cuda.synchronize(device)
            reset_if_needed(model)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        total_iterations = 0
        wall_start = time.perf_counter()
        with PowerMonitor(device=device, interval_s=sample_interval_s) as monitor:
            torch.cuda.synchronize(device)
            start.record()
            while True:
                for _ in range(repeat):
                    _ = model(batch)
                    reset_if_needed(model)
                total_iterations += repeat
                torch.cuda.synchronize(device)
                if time.perf_counter() - wall_start >= min_measure_seconds:
                    break
            end.record()
            torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    return {
        "idle_power_w": idle_power_w,
        "peak_power_w": monitor.peak_power_w,
        "mean_power_w": monitor.mean_power_w,
        "net_peak_power_w": monitor.peak_power_w - idle_power_w,
        "elapsed_ms_total": elapsed_ms,
        "elapsed_ms_per_batch": elapsed_ms / total_iterations,
        "iterations": total_iterations,
        "power_samples": len(monitor.samples),
        "power_trace": monitor.samples,
    }


def add_point_labels(ax, rows, x_col, y_col):
    for _, row in rows.iterrows():
        ax.annotate(
            row["display_name"],
            (row[x_col], row[y_col]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )


def save_plots(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#d62728" if row.model == "CUPY_SNN_LIF_READOUT" else "#1f77b4" for row in df.itertuples()]
    legend_handles = [
        Patch(facecolor="#1f77b4", edgecolor="#222222", label="Сверточные сети"),
        Patch(facecolor="#d62728", edgecolor="#222222", label="Предложенная SNN"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(df["parameters"], df["accuracy_percent"], s=86, c=colors, edgecolor="#222222", linewidth=0.8)
    add_point_labels(ax, df, "parameters", "accuracy_percent")
    ax.set_xscale("log")
    ax.set_xlabel("Обучаемые параметры, логарифмическая шкала")
    ax.set_ylabel("Точность на BNCI2014_001 (%)")
    ax.set_title("Точность в зависимости от числа параметров")
    ax.legend(handles=legend_handles, loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "bnci2014_001_accuracy_vs_parameters.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(df["peak_power_w"], df["accuracy_percent"], s=86, c=colors, edgecolor="#222222", linewidth=0.8)
    add_point_labels(ax, df, "peak_power_w", "accuracy_percent")
    ax.set_xlabel("Пиковая мощность GPU при инференсе (Вт)")
    ax.set_ylabel("Точность на BNCI2014_001 (%)")
    ax.set_title("Точность в зависимости от энергопотребления")
    ax.legend(handles=legend_handles, loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "bnci2014_001_accuracy_vs_peak_power.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar(df["display_name"], df["peak_power_w"], color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_ylabel("Пиковая мощность GPU при инференсе (Вт)")
    ax.set_title("Гистограмма пикового энергопотребления")
    ax.legend(handles=legend_handles, loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "bnci2014_001_peak_power_histogram.png", dpi=240)
    plt.close(fig)


def save_power_distribution_plots(trace_df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    if trace_df.empty:
        return

    model_order = [
        "ShallowConvNet",
        "DeepConvNet",
        "EEGNet",
        "Предложенная SNN",
    ]
    traces = [
        trace_df.loc[trace_df["display_name"] == name, "power_w"].to_numpy()
        for name in model_order
        if name in set(trace_df["display_name"])
    ]
    labels = [name for name in model_order if name in set(trace_df["display_name"])]
    colors = ["#d62728" if name == "Предложенная SNN" else "#1f77b4" for name in labels]
    legend_handles = [
        Patch(facecolor="#1f77b4", edgecolor="#222222", label="Сверточные сети"),
        Patch(facecolor="#d62728", edgecolor="#222222", label="Предложенная SNN"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    positions = range(1, len(traces) + 1)
    box = ax.boxplot(traces, positions=positions, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("Мощность GPU при инференсе (Вт)")
    ax.set_title("Распределение мощности GPU по моделям")
    ax.legend(handles=legend_handles, loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "bnci2014_001_power_distribution_boxplot.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for name, color in zip(labels, colors):
        values = trace_df.loc[trace_df["display_name"] == name, "power_w"].to_numpy()
        ax.hist(values, bins=18, alpha=0.34, density=True, color=color, label=name)
    ax.set_xlabel("Мощность GPU при инференсе (Вт)")
    ax.set_ylabel("Плотность")
    ax.set_title("Распределение замеров мощности GPU")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "bnci2014_001_power_distribution_hist.png", dpi=240)
    plt.close(fig)


def save_plots(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#d62728" if row.model == "CUPY_SNN_LIF_READOUT" else "#1f77b4" for row in df.itertuples()]
    legend_handles = [
        Patch(facecolor="#1f77b4", edgecolor="#222222", label=RU_CONV),
        Patch(facecolor="#d62728", edgecolor="#222222", label=RU_PROPOSED),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(df["parameters"], df["accuracy_percent"], s=86, c=colors, edgecolor="#222222", linewidth=0.8)
    add_point_labels(ax, df, "parameters", "accuracy_percent")
    ax.set_xscale("log")
    ax.set_xlabel(RU_TRAINABLE_PARAMS)
    ax.set_ylabel(RU_ACCURACY)
    ax.set_title(RU_ACC_PARAMS_TITLE)
    ax.legend(handles=legend_handles, loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "bnci2014_001_accuracy_vs_parameters.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(df["peak_power_w"], df["accuracy_percent"], s=86, c=colors, edgecolor="#222222", linewidth=0.8)
    add_point_labels(ax, df, "peak_power_w", "accuracy_percent")
    ax.set_xlabel(RU_PEAK_POWER)
    ax.set_ylabel(RU_ACCURACY)
    ax.set_title(RU_ACC_POWER_TITLE)
    ax.legend(handles=legend_handles, loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "bnci2014_001_accuracy_vs_peak_power.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar(df["display_name"], df["peak_power_w"], color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_ylabel(RU_PEAK_POWER)
    ax.set_title(RU_PEAK_HIST_TITLE)
    ax.legend(handles=legend_handles, loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "bnci2014_001_peak_power_histogram.png", dpi=240)
    plt.close(fig)


def save_power_distribution_plots(trace_df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    if trace_df.empty:
        return

    model_order = ["ShallowConvNet", "DeepConvNet", "EEGNet", RU_PROPOSED]
    traces = [
        trace_df.loc[trace_df["display_name"] == name, "power_w"].to_numpy()
        for name in model_order
        if name in set(trace_df["display_name"])
    ]
    labels = [name for name in model_order if name in set(trace_df["display_name"])]
    colors = ["#d62728" if name == RU_PROPOSED else "#1f77b4" for name in labels]
    legend_handles = [
        Patch(facecolor="#1f77b4", edgecolor="#222222", label=RU_CONV),
        Patch(facecolor="#d62728", edgecolor="#222222", label=RU_PROPOSED),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    box = ax.boxplot(traces, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel(RU_POWER)
    ax.set_title(RU_POWER_BY_MODEL)
    ax.legend(handles=legend_handles, loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "bnci2014_001_power_distribution_boxplot.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for name, color in zip(labels, colors):
        values = trace_df.loc[trace_df["display_name"] == name, "power_w"].to_numpy()
        ax.hist(values, bins=18, alpha=0.34, density=True, color=color, label=name)
    ax.set_xlabel(RU_POWER)
    ax.set_ylabel(RU_DENSITY)
    ax.set_title(RU_POWER_HIST_TITLE)
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "bnci2014_001_power_distribution_hist.png", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="BNCI2014_001 accuracy/parameters/power comparison")
    parser.add_argument("--moabb-csv", default="benchmarks/results/moabb/moabb_benchmark.csv")
    parser.add_argument("--proposed-json", default="benchmarks/results/full_eval/full_trials_plif_vs_lif_accuracy_table.json")
    parser.add_argument("--output-dir", default="benchmarks/results/model_efficiency")
    parser.add_argument("--time-step", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--min-measure-seconds", type=float, default=5.0)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--skip-power", action="store_true")
    args = parser.parse_args()

    accuracies = read_moabb_accuracy(args.moabb_csv)
    proposed_accuracy = read_proposed_accuracy(args.proposed_json)
    if proposed_accuracy is None:
        raise FileNotFoundError(f"Proposed SNN accuracy JSON not found: {args.proposed_json}")
    accuracies["CUPY_SNN_LIF_READOUT"] = proposed_accuracy

    if not args.skip_power and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for power measurement. Use --skip-power to only count parameters.")
    if not args.skip_power:
        torch.cuda.set_device(args.device)

    batch = torch.randn(args.batch_size, BNCI2014_001_CHANNELS, args.time_step)
    rows = []
    trace_rows = []
    for model_name in DEFAULT_MODELS:
        model = build_model(model_name, args.time_step, args.beta)
        row = {
            "model": model_name,
            "display_name": DISPLAY_NAMES[model_name],
            "accuracy_percent": accuracies[model_name],
            "parameters": count_parameters(model),
            "time_step": args.time_step,
            "batch_size": args.batch_size,
        }
        if args.skip_power:
            row.update(
                {
                    "idle_power_w": math.nan,
                    "peak_power_w": math.nan,
                    "mean_power_w": math.nan,
                    "net_peak_power_w": math.nan,
                    "elapsed_ms_total": math.nan,
                    "elapsed_ms_per_batch": math.nan,
                    "iterations": 0,
                    "power_samples": 0,
                }
            )
        else:
            power_result = benchmark_power(
                model=model,
                batch=batch,
                warmup=args.warmup,
                repeat=args.repeat,
                device=args.device,
                sample_interval_s=args.sample_interval,
                min_measure_seconds=args.min_measure_seconds,
            )
            for sample_idx, (timestamp_s, power_w) in enumerate(power_result.pop("power_trace")):
                trace_rows.append(
                    {
                        "model": model_name,
                        "display_name": DISPLAY_NAMES[model_name],
                        "sample_idx": sample_idx,
                        "timestamp_s": timestamp_s,
                        "power_w": power_w,
                    }
                )
            row.update(power_result)
        rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "bnci2014_001_params_power_metrics.csv"
    df.to_csv(metrics_path, index=False)
    trace_df = pd.DataFrame(trace_rows)
    trace_path = output_dir / "bnci2014_001_power_trace.csv"
    if not trace_df.empty:
        trace_df.to_csv(trace_path, index=False)
    if not args.skip_power:
        save_plots(df, output_dir / "plots")
        save_power_distribution_plots(trace_df, output_dir / "plots")

    print(f"Saved metrics: {metrics_path.resolve()}")
    if not trace_df.empty:
        print(f"Saved power trace: {trace_path.resolve()}")
    if not args.skip_power:
        print(f"Saved plots: {(output_dir / 'plots').resolve()}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
