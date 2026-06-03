"""Plot Weibo2014 benchmark metrics and model comparison summaries."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def parse_mean_std(cell: str):
    value = (cell or "").strip().strip('"')
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)±([0-9]+(?:\.[0-9]+)?)$", value)
    if not match:
        raise ValueError(f"Unexpected accuracy format: {cell!r}")
    return float(match.group(1)), float(match.group(2))


def parse_custom_accuracy(value: str, custom_std: float | None):
    text = (value or "").strip()
    if "±" in text:
        return parse_mean_std(text)
    mean = float(text)
    std = float(custom_std) if custom_std is not None else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Plot MOABB benchmark accuracies for Weibo2014")
    parser.add_argument("--csv", default="benchmarks/moabb/moabb_benchmark.csv", help="Path to benchmark CSV file")
    parser.add_argument("--output", default="benchmarks/moabb/plots/weibo2014_benchmark_accuracy.png", help="Output image path")
    parser.add_argument("--custom-acc", type=str, default=None, help="Optional custom accuracy: 'mean±std' or 'mean'")
    parser.add_argument("--custom-std", type=float, default=None, help="Optional std when --custom-acc is plain mean")
    parser.add_argument("--custom-label", type=str, default="proposed network", help="Label for custom accuracy")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    methods = []
    means = []
    stds = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "Weibo2014" not in (reader.fieldnames or []):
            raise KeyError("Column 'Weibo2014' was not found in the CSV file")

        for row in reader:
            pipeline = row["Pipeline"].strip().strip('"')
            mean, std = parse_mean_std(row["Weibo2014"])
            methods.append(pipeline)
            means.append(mean)
            stds.append(std)

    custom_mean = None
    custom_label = args.custom_label
    if args.custom_acc is not None:
        custom_mean, custom_std = parse_custom_accuracy(args.custom_acc, args.custom_std)
        methods.append(custom_label)
        means.append(float(custom_mean))
        stds.append(float(custom_std))

    items = sorted(zip(methods, means, stds), key=lambda x: x[1], reverse=True)
    methods, means, stds = map(list, zip(*items))

    plt.figure(figsize=(12, 8))
    colors = ["#1f77b4"] * len(methods)
    if custom_mean is not None and custom_label in methods:
        colors[methods.index(custom_label)] = "#d62728"

    plt.barh(methods, means, color=colors, alpha=0.9)
    plt.gca().invert_yaxis()
    plt.xlabel("Accuracy (%)")
    plt.title("MOABB Benchmark: Weibo2014")
    plt.xlim(0, 100)
    plt.grid(axis="x", linestyle="--", alpha=0.35)

    legend_handles = [Patch(facecolor="#1f77b4", label="MOABB benchmark")]
    if custom_mean is not None:
        legend_handles.append(Patch(facecolor="#d62728", label="proposed model"))
    plt.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
