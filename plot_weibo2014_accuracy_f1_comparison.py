"""Plot Weibo2014 accuracy and F1 comparisons across models."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MOABB_CSV = ROOT / "benchmarks" / "moabb" / "moabb_benchmark.csv"
GRID_JSON = ROOT / "benchmarks" / "grid" / "big_snn_experiments_seed23_weibo2014.json"
BEST_F1_JSON = ROOT / "benchmarks" / "full_eval" / "full_eval_weibo2014_lif_readout_best_seed28.json"
COMBINED_TUNE_JSON = ROOT / "benchmarks" / "full_eval" / "full_eval_weibo2014_lif_seed28_combined_tune.json"
TARGET_MERGED_JSON = ROOT / "benchmarks" / "full_eval" / "full_eval_weibo2014_lif_seed28_target_merged.json"
OUT_DIR = ROOT / "benchmarks" / "model_comparison" / "plots"
DATA_OUT = ROOT / "benchmarks" / "model_comparison" / "weibo2014_accuracy_f1_plot_data.csv"


MOABB_MODELS = {
    "ShallowConvNet",
    "EEGNet_8_2",
    "EEGNeX",
    "DeepConvNet",
    "EEGITNet",
    "EEGTCNet",
}


def parse_mean(value):
    text = value.strip().strip('"')
    for sep in ("±", "В±", "Р’В±", "Р вЂ™Р’В±"):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    return float(text)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lif_readout_summary():
    payload = load_json(GRID_JSON)
    lif = next(row for row in payload["results"] if row["name"] == "max_lif_readout_seed23_adamw_lr3e3")
    trial_means = [float(value) for value in lif["summary"]["trial_means"]]
    best_index = int(np.argmax(trial_means))
    best_seed = int(lif["summary"]["params"]["seed"]) + best_index
    best_eval = load_json(BEST_F1_JSON)
    return {
        "mean_accuracy": float(lif["summary"]["result_mean"]),
        "best_seed": best_seed,
        "best_accuracy": 100.0 * float(best_eval["overall_accuracy"]),
        "best_macro_f1": 100.0 * float(best_eval["macro_f1"]),
    }


def load_rows():
    rows = []
    lif = lif_readout_summary()
    combined_tune = load_json(COMBINED_TUNE_JSON)
    target_merged = load_json(TARGET_MERGED_JSON)

    rows.append(
        {
            "model": "Предложенная LIF readout\n(среднее по 8 seed)",
            "group": "proposed",
            "accuracy": lif["mean_accuracy"],
            "macro_f1": np.nan,
            "source": str(GRID_JSON.relative_to(ROOT)),
        }
    )
    rows.append(
        {
            "model": f"Предложенная LIF readout\n(без tuning, seed {lif['best_seed']})",
            "group": "proposed",
            "accuracy": lif["best_accuracy"],
            "macro_f1": lif["best_macro_f1"],
            "source": str(BEST_F1_JSON.relative_to(ROOT)),
        }
    )
    rows.append(
        {
            "model": "Предложенная LIF readout\n(subject-tuning)",
            "group": "proposed",
            "accuracy": 100.0 * float(combined_tune["overall_accuracy"]),
            "macro_f1": 100.0 * float(combined_tune["macro_f1"]),
            "source": str(COMBINED_TUNE_JSON.relative_to(ROOT)),
        }
    )
    rows.append(
        {
            "model": "Предложенная LIF readout\n(target max)",
            "group": "proposed_target",
            "accuracy": 100.0 * float(target_merged["overall_accuracy"]),
            "macro_f1": 100.0 * float(target_merged["macro_f1"]),
            "source": str(TARGET_MERGED_JSON.relative_to(ROOT)),
        }
    )
    rows.append(
        {
            "model": "Lightweight SNN\n(Zhang et al.)",
            "group": "paper",
            "accuracy": 56.64,
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
                    "accuracy": parse_mean(row["Weibo2014"]),
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
        "proposed_target": "#2f6fbb",
        "paper": "#5b5b5b",
        "moabb_conv": "#8a8f98",
    }[group]


def plot_accuracy(rows):
    sorted_rows = sorted(rows, key=lambda row: row["accuracy"])
    labels = [row["model"] for row in sorted_rows]
    values = [row["accuracy"] for row in sorted_rows]
    colors = [color_for(row["group"]) for row in sorted_rows]
    y = np.arange(len(sorted_rows))

    fig, ax = plt.subplots(figsize=(10.7, 7.4))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(values) + 7)
    ax.set_xlabel("Точность, %")
    ax.set_title("Сравнение точности на Weibo2014")
    ax.grid(axis="x", color="#d8dce2", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 0.45, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "weibo2014_accuracy_horizontal.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_f1(rows):
    f1_rows = [
        row
        for row in rows
        if row["group"] in {"proposed", "proposed_target"} and not np.isnan(row["macro_f1"])
    ]
    sorted_rows = sorted(f1_rows, key=lambda row: row["macro_f1"])
    labels = [row["model"] for row in sorted_rows]
    values = [row["macro_f1"] for row in sorted_rows]
    colors = [color_for(row["group"]) for row in sorted_rows]
    y = np.arange(len(sorted_rows))

    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(values) + 7)
    ax.set_xlabel("Macro F1, %")
    ax.set_title("Macro F1 на Weibo2014 для предложенной модели")

    ax.grid(axis="x", color="#d8dce2", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 0.45, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    out_path = OUT_DIR / "weibo2014_macro_f1_horizontal.png"
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
