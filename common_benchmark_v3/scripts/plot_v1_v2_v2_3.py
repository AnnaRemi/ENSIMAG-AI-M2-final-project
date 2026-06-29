#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMPLEMENTATIONS = [
    "trummer_heterogen_v1",
    "trummer_heterogen_v2_cascade",
    "trummer_heterogen_v2_3_batched_cascade",
]
LABELS = ["V1", "V2", "V2_3"]
CHEAP_COLOR = "#4C78A8"
EXPENSIVE_COLOR = "#F58518"
OVERHEAD_COLOR = "#B8B8B8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.outputs_dir)
    frame = pd.read_csv(output_dir / "all_metrics.csv")
    frame = (
        frame.set_index("implementation")
        .loc[IMPLEMENTATIONS]
        .reset_index()
    )

    plot_quality(frame, output_dir / "v1_v2_v2_3_quality.png")
    plot_calls(frame, output_dir / "v1_v2_v2_3_llm_calls.png")
    plot_wall_time(frame, output_dir / "v1_v2_v2_3_wall_time.png")


def finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, digits: int = 2) -> None:
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt=f"%.{digits}f",
            padding=3,
            fontsize=9,
        )


def plot_quality(frame: pd.DataFrame, path: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    colors = ["#54A24B", "#E45756", "#72B7B2"]
    x = np.arange(len(frame))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(
            x + (index - 1) * width,
            frame[metric],
            width,
            label=metric.capitalize(),
            color=color,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval quality: V1 vs V2 vs V2_3")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_calls(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cheap = frame["cheap_calls"].astype(float)
    expensive = frame["expensive_calls"].astype(float)
    cheap_bars = ax.bar(LABELS, cheap, color=CHEAP_COLOR, label="Cheap calls")
    expensive_bars = ax.bar(
        LABELS,
        expensive,
        bottom=cheap,
        color=EXPENSIVE_COLOR,
        label="Expensive calls",
    )
    ax.bar_label(cheap_bars, fmt="%.0f", label_type="center", color="white")
    ax.bar_label(
        expensive_bars,
        fmt="%.0f",
        label_type="center",
        color="black",
    )
    for index, total in enumerate(frame["llm_calls"]):
        ax.text(index, total + 0.8, f"Total {int(total)}", ha="center", fontsize=9)
    ax.set_ylabel("LLM calls")
    ax.set_title("LLM calls by model stage")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_wall_time(frame: pd.DataFrame, path: Path) -> None:
    wall = frame["wall_seconds"].astype(float)
    cheap = frame["cheap_seconds"].astype(float).clip(lower=0)
    expensive = frame["expensive_seconds"].astype(float).clip(lower=0)
    model_total = cheap + expensive

    # Stage timers can differ slightly from wall time. Scale only if their sum
    # exceeds wall time, preserving a valid wall-time decomposition.
    scale = np.where(model_total > wall, wall / model_total, 1.0)
    cheap_wall = cheap * scale
    expensive_wall = expensive * scale
    overhead = (wall - cheap_wall - expensive_wall).clip(lower=0)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cheap_bars = ax.bar(
        LABELS,
        cheap_wall,
        color=CHEAP_COLOR,
        label="Cheap-model time",
    )
    expensive_bars = ax.bar(
        LABELS,
        expensive_wall,
        bottom=cheap_wall,
        color=EXPENSIVE_COLOR,
        label="Expensive-model time",
    )
    overhead_bars = ax.bar(
        LABELS,
        overhead,
        bottom=cheap_wall + expensive_wall,
        color=OVERHEAD_COLOR,
        label="Other overhead",
    )
    for bars in (cheap_bars, expensive_bars, overhead_bars):
        labels = [f"{bar.get_height():.2f}" if bar.get_height() >= 0.05 else "" for bar in bars]
        ax.bar_label(bars, labels=labels, label_type="center", fontsize=8)
    for index, total in enumerate(wall):
        ax.text(index, total + max(wall) * 0.025, f"{total:.2f}s", ha="center")
    ax.set_ylabel("Wall seconds")
    ax.set_title("Wall time divided by model stage")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


if __name__ == "__main__":
    main()
