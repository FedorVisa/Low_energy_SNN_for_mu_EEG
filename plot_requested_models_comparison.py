"""Plot comparison charts for a requested subset of model results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Test accuracies (%) per subject from the executed runs on BNCI2014001 + 250Hz_preprocess_eeg.
    results_by_model = {
        "CUPY_SNN_PLIF": [81.94, 50.69, 81.60, 63.19, 51.74, 49.65, 72.92, 74.31, 76.04],
        "ShallowConvNet": [25.00, 26.04, 25.00, 27.08, 25.35, 24.65, 25.00, 25.00, 25.00],
        "deepconv": [25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00],
        "EEGNet": [30.56, 25.35, 25.69, 32.29, 23.61, 24.31, 25.00, 25.00, 25.00],
    }

    rows = []
    for model_name, values in results_by_model.items():
        arr = np.array(values, dtype=float)
        rows.append(
            {
                "Model": model_name,
                "MeanAcc": float(np.mean(arr)),
                "StdAcrossSubjects": float(np.std(arr)),
                "MinAcc": float(np.min(arr)),
                "MaxAcc": float(np.max(arr)),
                "NumSubjects": int(arr.size),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("MeanAcc", ascending=False).reset_index(drop=True)
    metrics_path = Path("benchmarks/results/model_comparison/requested_models_metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    labels = metrics_df["Model"].tolist()
    means = metrics_df["MeanAcc"].to_numpy()
    stds = metrics_df["StdAcrossSubjects"].to_numpy()
    colors = ["#d62728" if name == "CUPY_SNN_PLIF" else "#1f77b4" for name in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, yerr=stds, capsize=5, color=colors, ecolor="#333333", alpha=0.92)
    plt.ylabel("Accuracy (%)")
    plt.title("BNCI2014001 (250Hz_preprocess_eeg): test accuracy")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=15)
    plt.tight_layout()

    plot_path = Path("benchmarks/figures/model_comparison/requested_models_bnci2014001_250hz.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=220)

    print(f"Saved metrics: {metrics_path.resolve()}")
    print(f"Saved plot: {plot_path.resolve()}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
