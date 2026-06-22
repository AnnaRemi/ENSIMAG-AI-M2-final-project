#!/usr/bin/env python3
"""Generate the baseline vs online-join benchmark comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "benchmarks" / "9Q_comparison_medium_difficulty" / "metrics.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "benchmark_comparison.png"

METRICS = [
    ("wall_seconds", "Wall-clock Latency", "seconds"),
    ("engine_seconds", "Engine Time", "seconds"),
    ("llm_prompts", "LLM Prompts Issued", "count"),
    ("structured_candidates", "Structured Candidates", "count"),
    ("semantic_rows", "Semantic Rows Retrieved", "count"),
    ("join_rows", "Join Rows", "count"),
    ("result_rows", "Result Rows", "count"),
]

COLORS = {
    "baseline": "#238bb8",
    "online_join": "#f04455",
}


def query_label(query_id: str) -> str:
    number = query_id.split("_", 1)[0]
    return number.upper()


def add_value_labels(ax: plt.Axes, xs: list[str], ys: pd.Series, color: str) -> None:
    values = pd.to_numeric(ys, errors="coerce")
    if values.dropna().empty:
        return

    y_max = max(values.dropna().max(), 1)
    offset = y_max * 0.035
    for x, y in zip(xs, values):
        if pd.isna(y):
            continue
        ax.text(
            x,
            y + offset,
            f"{y:.0f}",
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )


def plot_metric(ax: plt.Axes, data: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    pivot = data.pivot(index="query_label", columns="project", values=metric)
    xs = list(pivot.index)

    for project in ("baseline", "online_join"):
        if project not in pivot:
            continue
        ys = pd.to_numeric(pivot[project], errors="coerce")
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            markersize=6,
            color=COLORS[project],
            label=project,
        )
        ax.fill_between(xs, ys.fillna(0), alpha=0.09, color=COLORS[project])
        add_value_labels(ax, xs, ys, COLORS[project])

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e9f0", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8)


def build_question_index(data: pd.DataFrame) -> str:
    questions = (
        data[["query_label", "question"]]
        .drop_duplicates()
        .sort_values("query_label")
        .itertuples(index=False)
    )
    return "\n".join(f"{label} - {question}" for label, question in questions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark comparison plot.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = pd.read_csv(args.metrics)
    data["query_label"] = data["query_id"].map(query_label)
    data = data.sort_values(["query_label", "project"])

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    axes = axes.flatten()
    fig.suptitle("Question Index", y=0.985, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.969,
        build_question_index(data),
        ha="center",
        va="top",
        fontsize=10,
        family="monospace",
    )

    for ax, (metric, title, ylabel) in zip(axes, METRICS):
        plot_metric(ax, data, metric, title, ylabel)

    axes[-1].axis("off")
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.87), h_pad=4.5, w_pad=3.0)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
