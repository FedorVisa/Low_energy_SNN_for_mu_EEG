"""Plot energy and parameter comparisons for BNCI2014-001 models."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROFILE_CSV = Path("bnci2014001_inference_profile.csv")
OUTPUT_DIR = Path("benchmarks/figures/model_efficiency")
OUTPUT_DATA = Path("benchmarks/results/model_efficiency/bnci2014_001_energy_params_plot_data.csv")

DISPLAY_NAMES = {
    "ShallowConvNet": "ShallowConvNet",
    "deepconv": "DeepConvNet",
    "EEGNet": "EEGNet",
    "CUPY_SNN_LIF_READOUT": "Предложенная LIF readout",
    "CUPY_SNN_PLIF": "Lightweight SNN (Zhang et al.)",
}

MODEL_ORDER = [
    "CUPY_SNN_PLIF",
    "CUPY_SNN_LIF_READOUT",
    "EEGNet",
    "ShallowConvNet",
    "deepconv",
]

COLORS = {
    "snn": "#2f73b8",
    "cnn": "#9198a1",
}


def read_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["model"].isin(DISPLAY_NAMES)].copy()
    df["display_name"] = df["model"].map(DISPLAY_NAMES)
    df["group"] = df["model"].apply(lambda name: "snn" if name.startswith("CUPY_SNN") else "cnn")
    df["order"] = df["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    return df.sort_values("order")


def add_bar_labels(ax, values, fmt):
    x_max = max(values)
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_width() + x_max * 0.015,
            patch.get_y() + patch.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=10,
        )


def save_horizontal_bar(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    value_fmt: str,
    log_scale: bool = False,
):
    plot_df = df.sort_values(value_col, ascending=True)
    colors = [COLORS[group] for group in plot_df["group"]]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(plot_df["display_name"], plot_df[value_col], color=colors, edgecolor="#222222", linewidth=0.6)
    add_bar_labels(ax, plot_df[value_col].to_list(), value_fmt)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    if log_scale:
        ax.set_xscale("log")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main():
    if not PROFILE_CSV.exists():
        raise FileNotFoundError(f"Не найден файл профилирования: {PROFILE_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_profile(PROFILE_CSV)
    df[
        [
            "model",
            "display_name",
            "group",
            "params_trainable",
            "energy_per_sample_mj",
            "avg_power_w",
            "peak_power_w",
            "inference_time_s",
            "throughput_samples_per_s",
        ]
    ].to_csv(OUTPUT_DATA, index=False)

    save_horizontal_bar(
        df=df,
        value_col="energy_per_sample_mj",
        output_path=OUTPUT_DIR / "bnci2014_001_energy_per_sample_horizontal.png",
        title="Энергия инференса на один пример на BNCI2014_001",
        xlabel="Энергия на пример, мДж",
        value_fmt="{:.2f}",
    )
    save_horizontal_bar(
        df=df,
        value_col="params_trainable",
        output_path=OUTPUT_DIR / "bnci2014_001_trainable_params_horizontal.png",
        title="Количество обучаемых параметров моделей",
        xlabel="Обучаемые параметры, логарифмическая шкала",
        value_fmt="{:,.0f}",
        log_scale=True,
    )

    print(f"Saved data: {OUTPUT_DATA.resolve()}")
    print(f"Saved plots: {OUTPUT_DIR.resolve()}")
    print(df[["display_name", "params_trainable", "energy_per_sample_mj"]].to_string(index=False))


if __name__ == "__main__":
    main()
