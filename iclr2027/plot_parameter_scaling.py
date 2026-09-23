#!/usr/bin/env python3
"""Plot exact trainable-parameter counts for the large-model stream sweep."""

from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VOCAB_SIZE = 50_304
BLOCK_SIZE = 1_024
NUM_LAYERS = 24
DIM = 1_024
NUM_HYPER_CONNECTIONS = 2 * NUM_LAYERS
STREAMS = np.array([4, 8, 16, 32])


def backbone_parameters() -> int:
    """GPT parameters shared by all methods, including position embeddings."""
    embeddings = VOCAB_SIZE * DIM + BLOCK_SIZE * DIM
    transformer_blocks = NUM_LAYERS * (12 * DIM**2 + 13 * DIM)
    final_layer_norm = 2 * DIM
    return embeddings + transformer_blocks + final_layer_norm


def parameters_per_hyper_connection(method: str, n: int) -> int:
    common = 2 * DIM * n**2 + DIM * n + 2 * n + 3
    if method == "ITO-HC":
        return common + n**2
    if method == "mHC (Sinkhorn)":
        return common + DIM * n**3 + n**2
    if method == "mHC-lite":
        return common + (DIM * n + 1) * factorial(n)
    raise ValueError(f"Unknown method: {method}")


def total_parameters(method: str, n: int) -> int:
    return backbone_parameters() + NUM_HYPER_CONNECTIONS * parameters_per_hyper_connection(method, n)


def compact_count(value: float) -> str:
    if value < 1e9:
        return f"{value / 1e6:.0f}M"
    if value < 1e12:
        return f"{value / 1e9:.1f}B"
    return f"{value:.2e}"


def main() -> None:
    methods = ["ITO-HC", "mHC (Sinkhorn)", "mHC-lite"]
    colors = ["#2A7F62", "#416788", "#C65D3B"]
    values = {
        method: np.array([total_parameters(method, int(n)) for n in STREAMS], dtype=float)
        for method in methods
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), gridspec_kw={"wspace": 0.30})
    x = np.arange(len(STREAMS))
    width = 0.24

    ax = axes[0]
    for index, (method, color) in enumerate(zip(methods, colors)):
        positions = x + (index - 1) * width
        bars = ax.bar(
            positions,
            values[method],
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=method,
            zorder=3,
        )
        if method == "mHC-lite":
            for bar, value in zip(bars[1:], values[method][1:]):
                ax.annotate(
                    compact_count(value),
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                    rotation=25,
                )
    ax.set_yscale("log")
    ax.set_ylim(1.5e8, 2e43)
    ax.set_xticks(x, STREAMS)
    ax.set_xlabel("Number of streams, $n$")
    ax.set_ylabel("Total trainable parameters")
    ax.set_title("(a) All methods (log scale)", loc="left")
    ax.grid(axis="y", which="major", color="#D7D7D7", linewidth=0.5, zorder=0)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for index, (method, color) in enumerate(zip(methods[:2], colors[:2])):
        positions = x + (index - 0.5) * 0.34
        bars = ax.bar(
            positions,
            values[method] / 1e6,
            width=0.34,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=method,
            zorder=3,
        )
        for bar, value in zip(bars, values[method] / 1e6):
            ax.annotate(
                f"{value:.0f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    ax.set_ylim(0, 2_260)
    ax.set_xticks(x, STREAMS)
    ax.set_xlabel("Number of streams, $n$")
    ax.set_ylabel("Total trainable parameters (millions)")
    ax.set_title("(b) Practical-range detail", loc="left")
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.5, zorder=0)
    ax.legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "large_model_parameter_scaling.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "large_model_parameter_scaling.png", dpi=220, bbox_inches="tight")

    print(f"Shared large-model backbone: {backbone_parameters():,}")
    for n in STREAMS:
        row = ", ".join(f"{method}={total_parameters(method, int(n)):,}" for method in methods)
        print(f"n={n}: {row}")


if __name__ == "__main__":
    main()
