#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FOCUS = [
    "trummer_heterogen_v1",
    "trummer_heterogen_v2_cascade",
    "trummer_heterogen_v2_2_structured_pruned",
    "trummer_heterogen_v2_3_batched_cascade",
    "trummer_heterogen_v3_pruned_cascade",
]
LABELS = {
    "cheap_block_join": "Block join\ncheap",
    "trummer_heterogen_v1": "Block join\nexpensive",
    "trummer_heterogen_v2_cascade": "Row-wise\ncascade",
    "trummer_heterogen_v2_2_structured_pruned": "Structured pruning\nblock join",
    "trummer_heterogen_v2_3_batched_cascade": "Batch-wise\ncascade",
    "trummer_heterogen_v3_pruned_cascade": "Structured pruning\ncascade",
}
QUALITY_COLORS = {
    "precision": "#2878B5",
    "recall": "#E07A1F",
    "f1": "#6A5ACD",
}
CHEAP_COLOR = "#4C9F70"
EXPENSIVE_COLOR = "#D55E00"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cheap block join against expensive block join and cascades."
    )
    parser.add_argument("--cheap-block-dir", required=True)
    parser.add_argument("--heterogen-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cheap_dir = Path(args.cheap_block_dir)
    heterogen_dir = Path(args.heterogen_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cheap = pd.read_csv(cheap_dir / "all_metrics.csv")
    heterogen = pd.read_csv(heterogen_dir / "all_metrics.csv")

    cheap_v1 = cheap.loc[cheap["implementation"].eq("trummer_heterogen_v1")].copy()
    if cheap_v1.empty:
        raise SystemExit(f"No V1 row found in {cheap_dir / 'all_metrics.csv'}")
    cheap_v1 = cheap_v1.iloc[[0]].copy()
    cheap_v1["implementation"] = "cheap_block_join"
    cheap_v1["label"] = "Cheap block join"
    cheap_v1["cheap_model"] = cheap_v1["model"]
    cheap_v1["expensive_model"] = ""
    cheap_v1["cheap_calls"] = cheap_v1["llm_calls"]
    cheap_v1["expensive_calls"] = 0.0
    cheap_v1["planned_cheap_calls"] = cheap_v1.get("planned_expensive_calls", cheap_v1["llm_calls"])
    cheap_v1["planned_expensive_calls"] = 0.0
    cheap_v1["cheap_seconds"] = cheap_v1["expensive_seconds"]
    cheap_v1["expensive_seconds"] = 0.0
    cheap_v1["cheap_time_percent"] = 100.0
    cheap_v1["expensive_time_percent"] = 0.0

    indexed = heterogen.set_index("implementation")
    missing = [name for name in FOCUS if name not in indexed.index]
    if missing:
        raise SystemExit("Missing implementations: " + ", ".join(missing))
    frame = pd.concat(
        [cheap_v1, indexed.loc[FOCUS].reset_index()],
        ignore_index=True,
    )
    frame["plot_label"] = [LABELS[name] for name in frame["implementation"]]
    frame["table_label"] = frame["plot_label"].str.replace("\n", " ", regex=False)
    frame = frame.fillna("")

    frame.to_csv(output_dir / "all_metrics.csv", index=False)
    write_summary(frame, output_dir / "summary.md")
    plot_quality(frame, output_dir / "metrics_precision_recall_f1.png")
    plot_time(frame, output_dir / "time_bar_plot.png")
    plot_calls(frame, output_dir / "calls_bar_plot.png")

    config = {
        "cheap_block_dir": str(cheap_dir),
        "heterogen_dir": str(heterogen_dir),
        "plotted_implementations": frame["implementation"].tolist(),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    print(frame.to_string(index=False), flush=True)


def write_summary(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "table_label",
        "model",
        "cheap_model",
        "expensive_model",
        "wall_seconds",
        "llm_calls",
        "cheap_calls",
        "expensive_calls",
        "final_answer_rows",
        "precision",
        "recall",
        "f1",
    ]
    headings = ["version", *columns[1:]]
    lines = [
        "# Cheap block join vs heterogen versions",
        "",
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join(["---"] * len(headings)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                f"{value:.4f}" if isinstance(value, float) else str(value)
                for value in row
            )
            + " |"
        )
    lines += [
        "",
        "Generated plots:",
        "",
        "- `metrics_precision_recall_f1.png`",
        "- `time_bar_plot.png`",
        "- `calls_bar_plot.png`",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_quality(frame: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(frame))
    width = 0.23
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for offset, metric in enumerate(["precision", "recall", "f1"]):
        values = frame[metric].astype(float).to_numpy()
        bars = ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=metric.upper() if metric == "f1" else metric.title(),
            color=QUALITY_COLORS[metric],
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(x, frame["plot_label"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Precision, recall, and F1")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_time(frame: pd.DataFrame, path: Path) -> None:
    labels = frame["plot_label"].tolist()
    wall = frame["wall_seconds"].astype(float).to_numpy()
    cheap = frame["cheap_seconds"].astype(float).clip(lower=0).to_numpy()
    expensive = frame["expensive_seconds"].astype(float).clip(lower=0).to_numpy()
    cheap_part, expensive_part, cheap_pct, expensive_pct = split_wall(
        wall,
        cheap,
        expensive,
    )
    fig, ax = plt.subplots(figsize=(13, 5.8))
    cheap_bars = ax.bar(labels, cheap_part, color=CHEAP_COLOR, label="Cheap-model time")
    expensive_bars = ax.bar(
        labels,
        expensive_part,
        bottom=cheap_part,
        color=EXPENSIVE_COLOR,
        label="Expensive-model time",
    )
    annotate_segments(ax, cheap_bars, cheap_part, cheap_pct, "cheap")
    annotate_segments(ax, expensive_bars, expensive_part, expensive_pct, "expensive")
    top_pad = max(wall) * 0.03 if len(wall) else 0.0
    for index, total in enumerate(wall):
        ax.text(index, total + top_pad, f"{total:.2f}s", ha="center", fontweight="bold")
    ax.set_ylim(0, max(wall) * 1.16 if len(wall) and max(wall) > 0 else 1)
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("Wall time split by cheap and expensive model-call time")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def split_wall(
    wall: np.ndarray,
    cheap_seconds: np.ndarray,
    expensive_seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_total = cheap_seconds + expensive_seconds
    cheap_pct = np.divide(
        cheap_seconds,
        model_total,
        out=np.zeros_like(cheap_seconds),
        where=model_total > 0,
    )
    expensive_pct = np.divide(
        expensive_seconds,
        model_total,
        out=np.ones_like(expensive_seconds),
        where=model_total > 0,
    )
    return wall * cheap_pct, wall * expensive_pct, cheap_pct * 100, expensive_pct * 100


def plot_calls(frame: pd.DataFrame, path: Path) -> None:
    labels = frame["plot_label"].tolist()
    cheap = frame["cheap_calls"].astype(float).to_numpy()
    expensive = frame["expensive_calls"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(13, 5.8))
    cheap_bars = ax.bar(labels, cheap, color=CHEAP_COLOR, label="Cheap-model calls")
    expensive_bars = ax.bar(
        labels,
        expensive,
        bottom=cheap,
        color=EXPENSIVE_COLOR,
        label="Expensive-model calls",
    )
    annotate_counts(ax, cheap_bars, cheap, "cheap")
    annotate_counts(ax, expensive_bars, expensive, "expensive")
    totals = cheap + expensive
    top_pad = max(totals) * 0.03 if len(totals) else 0.0
    for index, total in enumerate(totals):
        ax.text(index, total + top_pad, f"total {format_count(total)}", ha="center")
    ax.set_ylim(0, max(totals) * 1.16 if len(totals) and max(totals) > 0 else 1)
    ax.set_ylabel("LLM calls")
    ax.set_title("LLM calls split by cheap and expensive model calls")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def annotate_segments(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    percentages: np.ndarray,
    label: str,
) -> None:
    for bar, value, percent in zip(bars, values, percentages):
        if value <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{percent:.0f}%\n{label}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9,
        )


def annotate_counts(ax: plt.Axes, bars, values: np.ndarray, label: str) -> None:
    for bar, value in zip(bars, values):
        if value <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{format_count(value)}\n{label}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9,
        )


def format_count(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
