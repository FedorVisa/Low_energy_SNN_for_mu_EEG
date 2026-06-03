"""Render the LIF readout architecture diagram with parameter annotations."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("benchmarks/figures/model_efficiency")
VERTICAL_OUTPUT = OUTPUT_DIR / "cupy_snn_lif_readout_architecture_vertical_with_params.png"
HORIZONTAL_OUTPUT = OUTPUT_DIR / "cupy_snn_lif_readout_architecture_horizontal_presentation.png"

IN_CHANNELS = 22
OUT_CLASSES = 4
BETA = 2
TIME_STEP = 1000
CHANNELS = IN_CHANNELS * BETA
KERNEL = TIME_STEP // 32


def layer_params():
    return {
        "input": 0,
        "spatial_conv": IN_CHANNELS * CHANNELS,
        "temporal_conv": CHANNELS * KERNEL,
        "bn_t": CHANNELS * 2,
        "plif": 1,
        "linear": CHANNELS * OUT_CLASSES + OUT_CLASSES,
        "bn_readout": OUT_CLASSES * 2,
        "lif": 0,
        "mean": 0,
    }


def architecture_layers():
    params = layer_params()
    return [
        {
            "title": "Вход EEG",
            "lines": [
                f"N x {IN_CHANNELS} x {TIME_STEP}",
                "каналы x время",
                f"парам.: {params['input']}",
            ],
            "color": "#eef3f9",
        },
        {
            "title": "Conv1d C",
            "lines": [
                f"{IN_CHANNELS} -> {CHANNELS}",
                "kernel=1",
                f"N x {CHANNELS} x {TIME_STEP}",
                f"парам.: {params['spatial_conv']}",
            ],
            "color": "#dfeaf8",
        },
        {
            "title": "Depthwise\nConv1d T",
            "lines": [
                f"groups={CHANNELS}",
                f"kernel={KERNEL}",
                f"N x {CHANNELS} x {TIME_STEP}",
                f"парам.: {params['temporal_conv']}",
            ],
            "color": "#dfeaf8",
        },
        {
            "title": "BatchNorm1d",
            "lines": [
                f"{CHANNELS} каналов",
                "gamma + beta",
                f"N x {CHANNELS} x {TIME_STEP}",
                f"парам.: {params['bn_t']}",
            ],
            "color": "#dfeaf8",
        },
        {
            "title": "PLIF",
            "lines": [
                "форма: T x N x 44",
                "learnable tau",
                f"{TIME_STEP} x N x {CHANNELS}",
                f"парам.: {params['plif']}",
            ],
            "color": "#fff0c7",
        },
        {
            "title": "Linear",
            "lines": [
                f"{CHANNELS} -> {OUT_CLASSES}",
                "для каждого t",
                f"{TIME_STEP} x N x {OUT_CLASSES}",
                f"парам.: {params['linear']}",
            ],
            "color": "#dfeaf8",
        },
        {
            "title": "BatchNorm1d\nreadout",
            "lines": [
                f"{OUT_CLASSES} класса",
                "норм. логитов",
                f"{TIME_STEP} x N x {OUT_CLASSES}",
                f"парам.: {params['bn_readout']}",
            ],
            "color": "#dfeaf8",
        },
        {
            "title": "LIF readout",
            "lines": [
                "v_thr=0.2",
                "scale=2.5",
                f"{TIME_STEP} x N x {OUT_CLASSES}",
                f"парам.: {params['lif']}",
            ],
            "color": "#fff0c7",
        },
        {
            "title": "Среднее\nпо времени",
            "lines": [
                "mean(T)",
                f"логиты: N x {OUT_CLASSES}",
                f"парам.: {params['mean']}",
            ],
            "color": "#e3f2e8",
        },
    ]


def draw_box(ax, x, y, w, h, title, lines, facecolor, title_size=14, text_size=12, left=True):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.8,
        edgecolor="#223247",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    if left:
        tx = x + 0.22
        ha = "left"
    else:
        tx = x + w / 2
        ha = "center"
    ax.text(
        tx,
        y + h - 0.18,
        title,
        ha=ha,
        va="top",
        fontsize=title_size,
        fontweight="bold",
        linespacing=1.05,
    )
    ax.text(
        tx,
        y + h - 0.62,
        "\n".join(lines),
        ha=ha,
        va="top",
        fontsize=text_size,
        linespacing=1.15,
    )


def draw_arrow(ax, x0, y0, x1, y1):
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#46515f",
    )
    ax.add_patch(arrow)


def save_vertical(layers, total_params):
    fig, ax = plt.subplots(figsize=(9.4, 20.0))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 20.0)
    ax.axis("off")

    ax.text(4.6, 19.35, "Архитектура CUPY_SNN_LIF_READOUT", ha="center", va="center", fontsize=21, fontweight="bold")
    ax.text(
        4.6,
        18.88,
        f"BNCI2014_001: {IN_CHANNELS} каналов, {OUT_CLASSES} класса, T={TIME_STEP}, beta={BETA}. "
        f"Всего обучаемых параметров: {total_params}",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#3b4654",
    )

    box_x = 1.15
    box_w = 6.9
    box_h = 1.58
    gap = 0.28
    y = 16.85

    for idx, layer in enumerate(layers):
        draw_box(ax, box_x, y, box_w, box_h, layer["title"].replace("\n", " "), layer["lines"], layer["color"])
        if idx < len(layers) - 1:
            draw_arrow(ax, box_x + box_w / 2, y - 0.04, box_x + box_w / 2, y - gap + 0.05)
        y -= box_h + gap

    fig.tight_layout()
    fig.savefig(VERTICAL_OUTPUT, dpi=240)
    plt.close(fig)


def save_horizontal(layers, total_params):
    fig, ax = plt.subplots(figsize=(22.0, 7.0))
    ax.set_xlim(0, 22.0)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.text(11.0, 6.45, "Архитектура предложенной сети", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(
        11.0,
        6.05,
        f"BNCI2014_001: {IN_CHANNELS} каналов, {OUT_CLASSES} класса, T={TIME_STEP}, beta={BETA}. "
        f"Всего обучаемых параметров: {total_params}",
        ha="center",
        va="center",
        fontsize=13,
        color="#3b4654",
    )

    box_w = 2.08
    box_h = 3.15
    gap = 0.24
    y = 2.25
    x = 0.45
    centers = []

    for layer in layers:
        draw_box(
            ax,
            x,
            y,
            box_w,
            box_h,
            layer["title"],
            layer["lines"],
            layer["color"],
            title_size=13.5,
            text_size=12,
            left=False,
        )
        centers.append((x + box_w / 2, y + box_h / 2))
        x += box_w + gap

    for idx in range(len(layers) - 1):
        x0 = centers[idx][0] + box_w / 2 - 0.02
        x1 = centers[idx + 1][0] - box_w / 2 + 0.02
        draw_arrow(ax, x0, centers[idx][1], x1, centers[idx + 1][1])

    ax.text(
        11.0,
        0.8,
        "Dropout не добавляет параметров; LIF readout использует фиксированный tau.",
        ha="center",
        va="center",
        fontsize=12,
        color="#596273",
    )

    fig.tight_layout()
    fig.savefig(HORIZONTAL_OUTPUT, dpi=240)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_params = sum(layer_params().values())
    layers = architecture_layers()
    save_vertical(layers, total_params)
    save_horizontal(layers, total_params)
    print(f"Saved vertical architecture: {VERTICAL_OUTPUT.resolve()}")
    print(f"Saved horizontal architecture: {HORIZONTAL_OUTPUT.resolve()}")
    print(f"Total trainable params: {total_params}")


if __name__ == "__main__":
    main()
