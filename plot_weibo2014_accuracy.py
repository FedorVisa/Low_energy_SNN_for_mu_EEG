"""Plot Weibo2014 accuracy summaries from benchmark outputs."""

from pathlib import Path

import matplotlib.pyplot as plt


def main():
    model_name = "CUPY_SNN_PLIF"
    mean_acc = 52.6275
    std_acc = 0.4872917503918992

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        [model_name],
        [mean_acc],
        yerr=[std_acc],
        capsize=8,
        color="#d62728",
        ecolor="#333333",
        alpha=0.92,
    )
    plt.ylabel("Accuracy (%)")
    plt.title("Weibo2014 (250Hz_preprocess_eeg): test accuracy")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    label_text = f"{mean_acc:.4f}±{std_acc:.4f}"
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        plt.text(x, y + 1.2, label_text, ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = Path("benchmarks/figures/model_comparison/requested_models_weibo2014_250hz.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    print(f"Saved plot: {out_path.resolve()}")


if __name__ == "__main__":
    main()
