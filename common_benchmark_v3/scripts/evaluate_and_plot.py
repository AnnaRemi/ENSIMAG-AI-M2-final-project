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


ROOT = Path(__file__).resolve().parents[1]
FOCUS_IMPLEMENTATIONS = [
    "trummer_heterogen_v1",
    "trummer_heterogen_v2_cascade",
    "trummer_heterogen_v2_3_batched_cascade",
]
FOCUS_LABELS = {
    "suql_baseline": "SUQL\nbaseline",
    "trummer_heterogen_v1": "Block join",
    "trummer_heterogen_v2_cascade": "Row-wise cascade",
    "trummer_heterogen_v2_2_structured_pruned": "Structured pruning\nblock join",
    "trummer_heterogen_v2_3_batched_cascade": "Batch-wise cascade",
    "trummer_heterogen_v3_pruned_cascade": "Structured pruning\ncascade",
}
FALLBACK_FOCUS_ORDER = [
    "suql_baseline",
    "trummer_heterogen_v2_2_structured_pruned",
    "trummer_heterogen_v3_pruned_cascade",
    "trummer_heterogen_v1",
    "trummer_heterogen_v2_cascade",
    "trummer_heterogen_v2_3_batched_cascade",
]
QUALITY_COLORS = {
    "precision": "#2878B5",
    "recall": "#E07A1F",
    "f1": "#6A5ACD",
}
CHEAP_COLOR = "#4C9F70"
EXPENSIVE_COLOR = "#D55E00"
PLOT_FILES = {
    "metrics_precision_recall_f1.png",
    "time_bar_plot.png",
    "calls_bar_plot.png",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    args = parser.parse_args()
    outputs_dir = Path(args.outputs_dir)
    truth = set(json.loads((ROOT / "benchmark.json").read_text())["ground_truth_movie_ids"])
    rows = []
    outcomes = []
    for path in sorted(outputs_dir.glob("*/run_metrics.json")):
        run = json.loads(path.read_text())
        found = set(run["found_movie_ids"])
        tp, fp, fn, precision, recall, f1 = quality_from_run(run, found, truth)
        wall_seconds = float(run["wall_seconds"])
        cheap_calls = float(run.get("cheap_calls", 0))
        expensive_calls = float(run.get("expensive_calls", 0))
        cheap_seconds = float(run.get("cheap_seconds", 0.0))
        expensive_seconds = float(run.get("expensive_seconds", 0.0))
        cheap_time_percent = float(run.get("cheap_time_percent", 0.0))
        expensive_time_percent = float(run.get("expensive_time_percent", 0.0))
        if (
            expensive_calls > 0
            and cheap_calls == 0
            and cheap_seconds == 0
            and expensive_seconds == 0
        ):
            expensive_seconds = wall_seconds
            expensive_time_percent = 100.0
        rows.append({
            "implementation": run["implementation"],
            "mode": run["mode"],
            "model": run["model"],
            "cheap_model": run.get("cheap_model", ""),
            "expensive_model": run.get("expensive_model", ""),
            "wall_seconds": wall_seconds,
            "llm_calls": float(run["llm_calls"]),
            "cheap_calls": cheap_calls,
            "expensive_calls": expensive_calls,
            "cheap_seconds": cheap_seconds,
            "expensive_seconds": expensive_seconds,
            "cheap_time_percent": cheap_time_percent,
            "expensive_time_percent": expensive_time_percent,
            "cheap_early_accepts": float(run.get("cheap_early_accepts", 0)),
            "cheap_early_rejects": float(run.get("cheap_early_rejects", 0)),
            "expensive_candidates": float(run.get("expensive_candidates", 0)),
            "final_answer_rows": float(run["final_answer_rows"]),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        for movie_id in sorted(truth | found):
            outcomes.append({
                "implementation": run["implementation"],
                "movie_id": movie_id,
                "ground_truth": int(movie_id in truth),
                "found": int(movie_id in found),
                "classification": "TP" if movie_id in truth and movie_id in found else "FP" if movie_id in found else "FN",
            })
    if len(rows) < 2:
        raise SystemExit(f"Expected at least two run_metrics.json files under {outputs_dir}, found {len(rows)}")
    frame = pd.DataFrame(rows).sort_values("implementation")
    frame.to_csv(outputs_dir / "comparison.csv", index=False)
    pd.DataFrame(outcomes).to_csv(outputs_dir / "movie_id_outcomes.csv", index=False)
    write_markdown(frame, outputs_dir / "comparison.md")
    plot_requested(frame, outputs_dir)
    print(frame.to_string(index=False))


def quality_from_run(
    run: dict,
    found: set[str],
    truth: set[str],
) -> tuple[float, float, float, float, float, float]:
    if run.get("repetition_source") == "mean":
        return (
            float(run.get("true_positives", 0.0)),
            float(run.get("false_positives", 0.0)),
            float(run.get("false_negatives", 0.0)),
            float(run.get("precision", 0.0)),
            float(run.get("recall", 0.0)),
            float(run.get("f1", 0.0)),
        )
    tp = len(found & truth)
    fp = len(found - truth)
    fn = len(truth - found)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(tp), float(fp), float(fn), precision, recall, f1


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "implementation", "wall_seconds", "llm_calls", "cheap_calls",
        "cheap_early_accepts", "cheap_early_rejects", "expensive_calls",
        "cheap_time_percent", "expensive_time_percent",
        "precision", "recall", "f1",
    ]
    lines = [
        "# Trummer heterogen comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    indexed = frame.set_index("implementation")
    lines += [
        "",
        "SUQL applies structured SQL filters before calling answer() on the remaining reviews.",
        "The structured-pruned Trummer variants report their original input size and then apply deterministic year and ID pruning before semantic model calls.",
        "For cascade variants, `llm_calls = cheap_calls + expensive_calls`; non-cascade variants use expensive calls only.",
        "",
        "## Routing interpretation",
        "",
    ]
    if "trummer_heterogen_v1" in indexed.index:
        v1 = indexed.loc["trummer_heterogen_v1"]
        lines.append(f"- V1 issued {format_count(v1.llm_calls)} expensive block-join calls.")
    if "suql_baseline" in indexed.index:
        suql = indexed.loc["suql_baseline"]
        lines.append(
            f"- SUQL issued {format_count(suql.llm_calls)} answer() calls after structured SQL pruning."
        )
    if "trummer_heterogen_v2_2_structured_pruned" in indexed.index:
        v22 = indexed.loc["trummer_heterogen_v2_2_structured_pruned"]
        lines.append(
            f"- V2_2 issued {format_count(v22.llm_calls)} expensive block-join calls after deterministic pruning."
        )
    if "trummer_heterogen_v3_pruned_cascade" in indexed.index:
        v3 = indexed.loc["trummer_heterogen_v3_pruned_cascade"]
        lines += [
            f"- V3 issued {format_count(v3.cheap_calls)} cheap calls and {format_count(v3.expensive_calls)} expensive calls after deterministic pruning.",
            f"- V3 early-accepted {format_count(v3.cheap_early_accepts)} candidates and early-rejected "
            f"{format_count(v3.cheap_early_rejects)} candidates.",
            f"- {format_count(v3.expensive_candidates)} pruned candidates entered the V3 uncertainty band.",
        ]
    if "trummer_heterogen_v2_cascade" in indexed.index:
        v2 = indexed.loc["trummer_heterogen_v2_cascade"]
        lines += [
            f"- V2 issued {format_count(v2.cheap_calls)} cheap calls and {format_count(v2.expensive_calls)} expensive calls.",
            f"- V2 early-accepted {format_count(v2.cheap_early_accepts)} candidates and early-rejected "
            f"{format_count(v2.cheap_early_rejects)} candidates.",
            f"- {format_count(v2.expensive_candidates)} candidates entered the uncertainty band.",
            "- The cascade learns its cheap-decision confidence cutoff from an expensive-model "
            "calibration sample before final routing.",
        ]
    if "trummer_heterogen_v2_3_batched_cascade" in indexed.index:
        v23 = indexed.loc["trummer_heterogen_v2_3_batched_cascade"]
        lines += [
            f"- V2_3 issued {format_count(v23.cheap_calls)} batched cheap calls and "
            f"{format_count(v23.expensive_calls)} coalesced expensive calls.",
            f"- V2_3 model-call time: {v23.cheap_time_percent:.1f}% cheap and "
            f"{v23.expensive_time_percent:.1f}% expensive.",
        ]
    path.write_text("\n".join(lines) + "\n")


def plot_requested(frame: pd.DataFrame, outputs_dir: Path) -> None:
    focus = focus_frame(frame)
    clean_plot_outputs(outputs_dir)
    plot_metrics(focus, outputs_dir / "metrics_precision_recall_f1.png")
    plot_time(focus, outputs_dir / "time_bar_plot.png")
    plot_calls(focus, outputs_dir / "calls_bar_plot.png")


def focus_frame(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index("implementation")
    if all(item in indexed.index for item in FOCUS_IMPLEMENTATIONS):
        present = FOCUS_IMPLEMENTATIONS
    else:
        present = [item for item in FALLBACK_FOCUS_ORDER if item in indexed.index]
    if not present:
        raise SystemExit(f"No plottable implementations found in {frame}")
    focus = indexed.loc[present].copy()
    focus["plot_label"] = [FOCUS_LABELS[item] for item in present]
    return focus


def clean_plot_outputs(outputs_dir: Path) -> None:
    for path in outputs_dir.glob("*.png"):
        if path.name not in PLOT_FILES:
            path.unlink()


def finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(frame: pd.DataFrame, path: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    labels = frame["plot_label"].tolist()
    x = np.arange(len(frame))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, metric in enumerate(metrics):
        values = frame[metric].astype(float).to_numpy()
        bars = ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=metric.upper() if metric == "f1" else metric.title(),
            color=QUALITY_COLORS[metric],
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Precision, recall, and F1")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_time(frame: pd.DataFrame, path: Path) -> None:
    labels = frame["plot_label"].tolist()
    wall = frame["wall_seconds"].astype(float).to_numpy()
    cheap_seconds = frame["cheap_seconds"].astype(float).clip(lower=0).to_numpy()
    expensive_seconds = frame["expensive_seconds"].astype(float).clip(lower=0).to_numpy()
    cheap_part, expensive_part, cheap_pct, expensive_pct = split_wall_time(
        wall,
        cheap_seconds,
        expensive_seconds,
    )
    positive_wall = wall[wall > 0]
    if len(positive_wall) and max(positive_wall) / min(positive_wall) > 5:
        plot_time_horizontal(labels, wall, cheap_part, expensive_part, cheap_pct, expensive_pct, path)
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cheap_bars = ax.bar(labels, cheap_part, color=CHEAP_COLOR, label="Cheap-model time")
    expensive_bars = ax.bar(
        labels,
        expensive_part,
        bottom=cheap_part,
        color=EXPENSIVE_COLOR,
        label="Expensive-model time",
    )
    annotate_time_segments(ax, cheap_bars, cheap_part, cheap_pct, "cheap", "white")
    annotate_time_segments(
        ax,
        expensive_bars,
        expensive_part,
        expensive_pct,
        "expensive",
        "white",
    )
    top_pad = max(wall) * 0.03 if len(wall) else 0.0
    for index, total in enumerate(wall):
        ax.text(index, total + top_pad, f"{total:.2f}s", ha="center", fontweight="bold")
    ax.set_ylim(0, max(wall) * 1.16 if len(wall) and max(wall) > 0 else 1)
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("Wall time split by cheap and expensive model-call time")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_time_horizontal(
    labels: list[str],
    wall: np.ndarray,
    cheap_part: np.ndarray,
    expensive_part: np.ndarray,
    cheap_pct: np.ndarray,
    expensive_pct: np.ndarray,
    path: Path,
) -> None:
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    cheap_bars = ax.barh(
        y,
        cheap_part,
        color=CHEAP_COLOR,
        label="Cheap-model time",
    )
    expensive_bars = ax.barh(
        y,
        expensive_part,
        left=cheap_part,
        color=EXPENSIVE_COLOR,
        label="Expensive-model time",
    )
    xmax = max(wall) * 1.18 if len(wall) and max(wall) > 0 else 1
    label_x = xmax * 0.54
    for index, total in enumerate(wall):
        if cheap_part[index] > 0 and expensive_part[index] > 0:
            split_label = f"{cheap_pct[index]:.0f}% cheap / {expensive_pct[index]:.0f}% expensive"
        elif cheap_part[index] > 0:
            split_label = f"{cheap_pct[index]:.0f}% cheap"
        else:
            split_label = f"{expensive_pct[index]:.0f}% expensive"
        ax.text(
            label_x,
            index,
            split_label,
            va="center",
            ha="left",
            color="white" if total > label_x else "black",
            fontweight="bold",
            bbox=(
                None
                if total > label_x
                else {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5}
            ),
        )
        ax.text(
            total + xmax * 0.012,
            index,
            f"{total:.2f}s",
            va="center",
            fontweight="bold",
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Wall time (seconds)")
    ax.set_title("Wall time split by cheap and expensive model-call time")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    finish(fig, path)


def split_wall_time(
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


def annotate_time_segments(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    percentages: np.ndarray,
    label: str,
    color: str,
) -> None:
    for bar, value, percent in zip(bars, values, percentages):
        if value <= 0:
            continue
        text = f"{percent:.0f}%\n{label}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            text,
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            fontsize=9,
        )


def annotate_horizontal_time_segments(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    lefts: np.ndarray,
    percentages: np.ndarray,
    label: str,
    xmax: float,
) -> None:
    for bar, value, left, percent in zip(bars, values, lefts, percentages):
        if value <= 0:
            continue
        text = f"{percent:.0f}% {label}"
        if value >= xmax * 0.1:
            ax.text(
                left + value / 2,
                bar.get_y() + bar.get_height() / 2,
                text,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=9,
            )
        else:
            ax.text(
                left + value + xmax * 0.012,
                bar.get_y() + bar.get_height() / 2,
                text,
                ha="left",
                va="center",
                color="black",
                fontsize=9,
            )


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
    annotate_call_segments(ax, cheap_bars, cheap, "cheap", "white")
    annotate_call_segments(ax, expensive_bars, expensive, "expensive", "white")
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


def annotate_call_segments(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    label: str,
    color: str,
) -> None:
    for bar, value in zip(bars, values):
        if value <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{format_count(value)}\n{label}",
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            fontsize=9,
        )


def format_count(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


if __name__ == "__main__":
    main()
