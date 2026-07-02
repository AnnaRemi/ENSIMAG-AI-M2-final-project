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
LABELS = {
    "trummer_heterogen_v1": "V1\nBlock join",
    "trummer_heterogen_v2_cascade": "V2\nPair-wise cascade",
    "trummer_heterogen_v2_3_batched_cascade": "V2_3\nBatched cascade",
}
CHEAP_COLOR = "#4C9F70"
EXPENSIVE_COLOR = "#D55E00"
OVERHEAD_COLOR = "#B8B8B8"
QUALITY_COLORS = {
    "precision": "#2878B5",
    "recall": "#E07A1F",
    "f1": "#6A5ACD",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="All-Heterogen output directory containing all_metrics.csv.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Plot the available subset when V1, V2, or V2_3 is missing.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    frame = pd.read_csv(run_dir / "all_metrics.csv").set_index("implementation")
    missing = [name for name in IMPLEMENTATIONS if name not in frame.index]
    if missing and not args.allow_missing:
        raise SystemExit(f"Missing implementations: {', '.join(missing)}")
    present = [name for name in IMPLEMENTATIONS if name in frame.index]
    if not present:
        raise SystemExit("No V1/V2/V2_3 implementations found in all_metrics.csv")
    frame = frame.loc[present].copy()
    frame["plot_label"] = [LABELS[name] for name in present]
    title_suffix = " vs ".join(label.split("\n", 1)[0] for label in frame["plot_label"])

    plot_quality(frame, run_dir / "v1_v2_v2_3_quality.png", title_suffix)
    plot_calls(frame, run_dir / "v1_v2_v2_3_llm_calls.png")
    plot_wall_time(frame, run_dir / "v1_v2_v2_3_wall_time.png")


def plot_quality(frame: pd.DataFrame, path: Path, title_suffix: str) -> None:
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(frame))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for index, metric in enumerate(metrics):
        values = frame[metric].astype(float).to_numpy()
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=metric.upper() if metric == "f1" else metric.title(),
            color=QUALITY_COLORS[metric],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(value, 0.015),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x, frame["plot_label"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Retrieval quality: {title_suffix}")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_calls(frame: pd.DataFrame, path: Path) -> None:
    labels = frame["plot_label"].tolist()
    cheap = frame["cheap_calls"].astype(float).to_numpy()
    expensive = frame["expensive_calls"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    cheap_bars = ax.bar(labels, cheap, color=CHEAP_COLOR, label="Cheap-model calls")
    expensive_bars = ax.bar(
        labels,
        expensive,
        bottom=cheap,
        color=EXPENSIVE_COLOR,
        label="Expensive-model calls",
    )
    for index, (cheap_value, expensive_value) in enumerate(zip(cheap, expensive)):
        if cheap_value:
            ax.text(
                index,
                cheap_value / 2,
                f"{int(cheap_value)} cheap",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
        if expensive_value:
            ax.text(
                index,
                cheap_value + expensive_value / 2,
                f"{int(expensive_value)} expensive",
                ha="center",
                va="center",
                color="white" if expensive_value >= 2 else "black",
                fontweight="bold",
                fontsize=11 if expensive_value >= 2 else 9,
            )
        ax.text(
            index,
            cheap_value + expensive_value + 0.8,
            f"total {int(cheap_value + expensive_value)}",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("LLM calls")
    ax.set_title("LLM calls by model stage")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_wall_time(frame: pd.DataFrame, path: Path) -> None:
    labels = frame["plot_label"].tolist()
    wall = frame["wall_seconds"].astype(float).to_numpy()
    cheap = frame["cheap_seconds"].astype(float).to_numpy()
    expensive = frame["expensive_seconds"].astype(float).to_numpy()
    overhead = np.maximum(wall - cheap - expensive, 0.0)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(labels, cheap, color=CHEAP_COLOR, label="Cheap-model time")
    ax.bar(
        labels,
        expensive,
        bottom=cheap,
        color=EXPENSIVE_COLOR,
        label="Expensive-model time",
    )
    ax.bar(
        labels,
        overhead,
        bottom=cheap + expensive,
        color=OVERHEAD_COLOR,
        label="Other/overhead time",
    )
    for index, (total, cheap_value, expensive_value, overhead_value) in enumerate(
        zip(wall, cheap, expensive, overhead)
    ):
        if cheap_value >= 1:
            ax.text(
                index,
                cheap_value / 2,
                f"{cheap_value:.2f}s cheap",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=9,
            )
        if expensive_value >= 1:
            ax.text(
                index,
                cheap_value + expensive_value / 2,
                f"{expensive_value:.2f}s expensive",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=9,
            )
        if overhead_value >= 1:
            ax.text(
                index,
                cheap_value + expensive_value + overhead_value / 2,
                f"{overhead_value:.2f}s overhead",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
        ax.text(
            index,
            total + max(wall) * 0.025,
            f"{total:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylim(0, max(wall) * 1.15)
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("Wall time divided by cheap, expensive, and overhead time")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
