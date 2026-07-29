#!/usr/bin/env python3
"""Compare quality (precision/recall/F1) across model pairs, per dataset.

Reads each <dataset>_5q_10rep/<pair>/aggregate.csv and plots the four
methods side by side, grouped by model pair, so pairs can be compared
directly instead of only within their own subdirectory.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DATASETS = {
    "imdb": ROOT / "imdb_5q_10rep",
    "amazon": ROOT / "amazon_5q_10rep",
}

PAIR_ORDER = [
    "gemma4_e2b__gemma4_26b",
    "qwen3.6_27b__qwen3.6_35b",
    "gemma4_e2b__qwen3.6_35b_crossfamily",
    "llama3.1_8b__llama3.1_70b",
]
PAIR_LABELS = {
    "gemma4_e2b__gemma4_26b": "gemma4:e2b /\ngemma4:26b",
    "qwen3.6_27b__qwen3.6_35b": "qwen3.6:27b /\nqwen3.6:35b",
    "gemma4_e2b__qwen3.6_35b_crossfamily": "gemma4:e2b /\nqwen3.6:35b (cross)",
    "llama3.1_8b__llama3.1_70b": "llama3.1:8b /\nllama3.1:70b",
}

METHOD_ORDER = [
    "suql_baseline",
    "suql_v1_two_level_cascade",
    "trummer_baseline_adaptive_block_join",
    "trummer_v1_structured_two_level_cascade",
]
METHOD_SHORT = {
    "suql_baseline": "SUQL baseline",
    "suql_v1_two_level_cascade": "SUQL V1",
    "trummer_baseline_adaptive_block_join": "Trummer baseline",
    "trummer_v1_structured_two_level_cascade": "Trummer V1",
}
METHOD_COLORS = {
    "suql_baseline": "#7AA6C2",
    "suql_v1_two_level_cascade": "#315A7D",
    "trummer_baseline_adaptive_block_join": "#E8A36A",
    "trummer_v1_structured_two_level_cascade": "#A64B20",
}
QUALITY_METRICS = ["precision", "recall", "f1"]

# Amazon's aggregate.csv uses short method identifiers; IMDb uses the full ones.
METHOD_ALIASES = {
    "suql_baseline": "suql_baseline",
    "suql_v1": "suql_v1_two_level_cascade",
    "trummer_baseline": "trummer_baseline_adaptive_block_join",
    "trummer_v1": "trummer_v1_structured_two_level_cascade",
}


def load_dataset(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for pair in PAIR_ORDER:
        aggregate_path = dataset_dir / pair / "aggregate.csv"
        if not aggregate_path.exists():
            continue
        frame = pd.read_csv(aggregate_path)
        if "implementation" not in frame.columns:
            frame = frame.rename(columns={"method": "implementation"})
        frame["implementation"] = frame["implementation"].map(lambda value: METHOD_ALIASES.get(value, value))
        frame["pair"] = pair
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    combined["pair"] = pd.Categorical(combined["pair"], PAIR_ORDER, ordered=True)
    combined["implementation"] = pd.Categorical(combined["implementation"], METHOD_ORDER, ordered=True)
    return combined.sort_values(["pair", "implementation"])


def plot_quality_by_pair(frame: pd.DataFrame, dataset_label: str, path: Path) -> None:
    pair_labels = [PAIR_LABELS[pair] for pair in PAIR_ORDER]
    x = np.arange(len(PAIR_ORDER))
    width = 0.19
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True)
    for ax, metric in zip(axes, QUALITY_METRICS):
        pivot = frame.pivot(index="pair", columns="implementation", values=metric).reindex(PAIR_ORDER)
        for index, method in enumerate(METHOD_ORDER):
            values = pivot[method].to_numpy(float)
            ax.bar(
                x + (index - 1.5) * width, values, width,
                label=METHOD_SHORT[method], color=METHOD_COLORS[method],
            )
        ax.set_title(metric.title())
        ax.set_xticks(x, pair_labels, fontsize=8.5)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Score (higher is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{dataset_label}: quality by model pair and method", y=1.1, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_f1_pair_heatmap(frames: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(6.5 * len(frames), 5.6), sharey=True)
    if len(frames) == 1:
        axes = [axes]
    for index, (ax, (dataset_label, frame)) in enumerate(zip(axes, frames.items())):
        pivot = frame.pivot(index="pair", columns="implementation", values="f1").reindex(
            index=PAIR_ORDER, columns=METHOD_ORDER,
        )
        image = ax.imshow(pivot.to_numpy(float), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(METHOD_ORDER)), [METHOD_SHORT[method] for method in METHOD_ORDER], rotation=25, ha="right")
        if index == 0:
            ax.set_yticks(range(len(PAIR_ORDER)), [PAIR_LABELS[pair].replace("\n", " ") for pair in PAIR_ORDER], fontsize=8.5)
        for row in range(len(PAIR_ORDER)):
            for col in range(len(METHOD_ORDER)):
                value = pivot.to_numpy(float)[row, col]
                ax.text(
                    col, row, f"{value:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if value > 0.55 else "#111111",
                )
        ax.set_title(dataset_label)
    fig.suptitle("F1 across model pairs and methods", fontsize=15)
    fig.subplots_adjust(wspace=0.35, right=0.88)
    colorbar_axes = fig.add_axes((0.91, 0.15, 0.018, 0.7))
    fig.colorbar(image, cax=colorbar_axes, label="F1 (higher is better)")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for dataset_key, dataset_dir in DATASETS.items():
        frame = load_dataset(dataset_dir)
        dataset_label = "IMDb" if dataset_key == "imdb" else "Amazon Fashion"
        frames[dataset_label] = frame
        plot_quality_by_pair(frame, dataset_label, HERE / f"{dataset_key}_quality_by_pair.png")
    plot_f1_pair_heatmap(frames, HERE / "f1_by_pair_heatmap.png")


if __name__ == "__main__":
    main()
