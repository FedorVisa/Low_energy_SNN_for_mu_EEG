"""Plot accuracy and F1 comparisons for BNCI2014-001 model results."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MOABB_CSV = ROOT / "benchmarks" / "moabb" / "moabb_benchmark.csv"
OUT_DIR = ROOT / "benchmarks" / "model_comparison" / "plots"
DATA_OUT = ROOT / "benchmarks" / "model_comparison" / "bnci2014_001_accuracy_f1_plot_data.csv"


PROPOSED = [
    (
        "Предложенная LIF readout\n(без донастройки)",
        ROOT / "benchmarks" / "full_eval" / "full_eval_lif_readout_best_seed26.json",
        "proposed",
    ),
    (
        "Предложенная LIF readout\n(донастройка по субъектам)",
        ROOT / "benchmarks" / "full_eval" / "full_eval_subject_tuning_lif_seed26_selected.json",
        "proposed",
    ),
    (
        "Предложенная LIF readout\n(выбор максимума)",
        ROOT / "benchmarks" / "full_eval" / "full_eval_subject_tuning_lif_seed26_final_merged.json",
        "proposed",
    ),
]

MOABB_MODELS = {
    "ShallowConvNet",
    "EEGNet_8_2",
    "EEGNeX",
    "DeepConvNet",
    "EEGITNet",
    "EEGTCNet",
}


def percent_from_json(path, key):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return 100.0 * float(payload[key])


def parse_mean(value):
    text = value.strip().strip('"')
    for sep in ("±", "В±", "Р’В±"):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    return float(text)


def load_rows():
    rows = []
    for label, path, group in PROPOSED:
        rows.append(
            {
                "model": label,
                "group": group,
                "accuracy": percent_from_json(path, "overall_accuracy"),
                "macro_f1": percent_from_json(path, "macro_f1"),
                "source": str(path.relative_to(ROOT)),
            }
        )

    rows.append(
        {
            "model": "Lightweight SNN\n(Zhang et al.)",
            "group": "paper",
            "accuracy": 68.92,
            "macro_f1": np.nan,
            "source": "A lightweight spiking neural network for EEG-based motor imagery classification",
        }
    )

    with MOABB_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["Pipeline"]
            if model not in MOABB_MODELS:
                continue
            rows.append(
                {
                    "model": model,
                    "group": "moabb_conv",
                    "accuracy": parse_mean(row["BNCI2014_001"]),
                    "macro_f1": np.nan,
                    "source": str(MOABB_CSV.relative_to(ROOT)),
                }
            )
    return rows


def write_data(rows):
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DATA_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "group", "accuracy", "macro_f1", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def color_for(group):
    return {
        "proposed": "#2f6fbb",
        "paper": "#5b5b5b",
        "moabb_conv": "#8a8f98",
    }[group]


def plot_accuracy(rows):
    sorted_rows = sorted(rows, key=lambda row: row["accuracy"])
    labels = [row["model"] for row in sorted_rows]
    values = [row["accuracy"] for row in sorted_rows]
    colors = [color_for(row["group"]) for row in sorted_rows]
    y = np.arange(len(sorted_rows))

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(values) + 7)
    ax.set_xlabel("Точность, %")
    ax.set_title("Сравнение точности на BNCI2014_001")
    ax.grid(axis="x", color="#d8dce2", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 0.45, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "bnci2014_001_accuracy_horizontal.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_f1(rows):
    f1_rows = [row for row in rows if row["group"] == "proposed" and not np.isnan(row["macro_f1"])]
    sorted_rows = sorted(f1_rows, key=lambda row: row["macro_f1"])
    labels = [row["model"] for row in sorted_rows]
    values = [row["macro_f1"] for row in sorted_rows]
    y = np.arange(len(sorted_rows))

    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    bars = ax.barh(y, values, color="#2f6fbb", edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(values) + 7)
    ax.set_xlabel("Macro F1, %")
    ax.set_title("Macro F1 на BNCI2014_001 для вариантов предложенной модели")
    ax.grid(axis="x", color="#d8dce2", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 0.45, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "bnci2014_001_macro_f1_horizontal.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    write_data(rows)
    accuracy_path = plot_accuracy(rows)
    f1_path = plot_f1(rows)
    print(f"wrote {DATA_OUT}")
    print(f"wrote {accuracy_path}")
    print(f"wrote {f1_path}")


if __name__ == "__main__":
    main()
