"""Plot BNCI2014-001 experiment summaries and comparison figures."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def parse_mean_std(cell: str):
    """Parse strings like '72.38±14.85' into float mean and std."""
    value = (cell or "").strip().strip('"')
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)±([0-9]+(?:\.[0-9]+)?)$", value)
    if not match:
        raise ValueError(f"Unexpected accuracy format: {cell!r}")
    return float(match.group(1)), float(match.group(2))


def parse_custom_accuracy(value: str, custom_std: float | None):
    """Parse custom accuracy from 'mean±std' or 'mean' (+ optional --custom-std)."""
    text = (value or "").strip()
    if "±" in text:
        return parse_mean_std(text)
    mean = float(text)
    std = float(custom_std) if custom_std is not None else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Plot BNCI2014_001 benchmark accuracies")
    parser.add_argument(
        "--csv",
        default="benchmarks/moabb/moabb_benchmark.csv",
        help="Path to moabb benchmark CSV file",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/moabb/plots/bnci2014_001_accuracy.png",
        help="Output image path",
    )
    parser.add_argument(
        "--custom-acc",
        type=str,
        default=None,
        help="Optional custom accuracy: 'mean±std' or just 'mean'",
    )
    parser.add_argument(
        "--custom-std",
        type=float,
        default=None,
        help="Optional custom std when --custom-acc is provided as plain mean",
    )
    parser.add_argument(
        "--custom-label",
        type=str,
        default="proposed model",
        help="Label for first custom accuracy",
    )
    parser.add_argument(
        "--custom-acc-2",
        type=str,
        default=None,
        help="Optional second custom accuracy: 'mean±std' or just 'mean'",
    )
    parser.add_argument(
        "--custom-std-2",
        type=float,
        default=None,
        help="Optional std when --custom-acc-2 is provided as plain mean",
    )
    parser.add_argument(
        "--custom-label-2",
        type=str,
        default="proposed network",
        help="Label for second custom accuracy",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    methods = []
    means = []
    stds = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "BNCI2014_001" not in (reader.fieldnames or []):
            raise KeyError("Column 'BNCI2014_001' was not found in the CSV file")

        for row in reader:
            pipeline = row["Pipeline"].strip().strip('"')
            mean, std = parse_mean_std(row["BNCI2014_001"])
            methods.append(pipeline)
            means.append(mean)
            stds.append(std)

    custom_labels = set()
    if args.custom_acc is not None:
        custom_mean, custom_std = parse_custom_accuracy(args.custom_acc, args.custom_std)
        methods.append(args.custom_label)
        means.append(float(custom_mean))
        stds.append(float(custom_std))
        custom_labels.add(args.custom_label)

    if args.custom_acc_2 is not None:
        custom_mean_2, custom_std_2 = parse_custom_accuracy(args.custom_acc_2, args.custom_std_2)
        methods.append(args.custom_label_2)
        means.append(float(custom_mean_2))
        stds.append(float(custom_std_2))
        custom_labels.add(args.custom_label_2)

    # Sort descending to make comparison easier.
    items = sorted(zip(methods, means, stds), key=lambda x: x[1], reverse=True)
    methods, means, stds = map(list, zip(*items))

    plt.figure(figsize=(12, 8))
    colors = ["#1f77b4"] * len(methods)
    if custom_labels:
        for idx, method in enumerate(methods):
            if method in custom_labels:
                colors[idx] = "#d62728"

    plt.barh(methods, means, color=colors, alpha=0.9)
    plt.gca().invert_yaxis()
    plt.xlabel("Accuracy (%)")
    plt.title("MOABB Benchmark: BNCI2014_001")
    plt.xlim(0, 100)
    plt.grid(axis="x", linestyle="--", alpha=0.35)

    legend_handles = [Patch(facecolor="#1f77b4", label="MOABB benchmark")]
    if custom_labels:
        legend_handles.append(Patch(facecolor="#d62728", label="proposed network"))
    plt.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
