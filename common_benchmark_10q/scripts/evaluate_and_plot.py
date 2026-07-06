#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import textwrap

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common import ROOT


LABELS = {
    "suql_baseline": "SUQL",
    "suql_stage2_bargain_cascade": "SUQL_STAGE2",
    "trummer_heterogen_v2_3_batched_cascade": "V2_3",
    "trummer_heterogen_v3_pruned_cascade": "V3",
    "trummer_heterogen_v3_2_pruned_batched_cascade": "V3_2",
}
DISPLAY_LABELS = {
    "SUQL": "SUQL baseline",
    "SUQL_STAGE2": "SUQL Stage 2\nBARGAIN cascade",
    "V2_3": "Heterogen 2.3\nbatch cascade",
    "V3": "Heterogen 3\nstructured row cascade",
    "V3_2": "Heterogen 3.2\nstructured batch cascade",
}
SHORT_DISPLAY_LABELS = {
    "SUQL": "SUQL baseline",
    "SUQL_STAGE2": "SUQL Stage 2",
    "V2_3": "Heterogen 2.3",
    "V3": "Heterogen 3",
    "V3_2": "Heterogen 3.2",
}
METHOD_ORDER = ["SUQL", "SUQL_STAGE2", "V2_3", "V3", "V3_2"]
METHOD_CONTEXT = {
    "SUQL": "SUQL baseline: structured SQL + direct expensive-model semantic evaluation",
    "SUQL_STAGE2": "SUQL Stage 2: structured SQL + learned cheap-to-expensive cascade",
    "V2_3": "Heterogen 2.3: batch-wise cascade without SUQL-style structured pruning",
    "V3": "Heterogen 3: SUQL-style structured pruning + row-wise cascade",
    "V3_2": "Heterogen 3.2: SUQL-style structured pruning + batch-wise cascade",
}
COLORS = {
    "SUQL": "#2878B5",
    "SUQL_STAGE2": "#D55E00",
    "V2_3": "#8E5EA2",
    "V3": "#3A9D5D",
    "V3_2": "#4C9F70",
}
QUALITY_COLORS = {
    "Precision": "#4c78a8",
    "Recall": "#f58518",
    "F1": "#6f5bd3",
}
CHEAP_COLOR = "#4c9f70"
EXPENSIVE_COLOR = "#d95f02"


def load_questions() -> list[dict]:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    questions = manifest.get("questions", [])
    if len(questions) != 10:
        raise RuntimeError(f"Expected 10 questions in manifest, found {len(questions)}")
    return questions


def quality_from_run(run: dict, truth: set[str]) -> dict[str, float]:
    if all(key in run for key in ("precision", "recall", "f1")):
        return {
            "true_positives": float(run.get("true_positives", 0.0)),
            "false_positives": float(run.get("false_positives", 0.0)),
            "false_negatives": float(run.get("false_negatives", 0.0)),
            "precision": float(run["precision"]),
            "recall": float(run["recall"]),
            "f1": float(run["f1"]),
        }
    found = set(run["found_movie_ids"])
    tp = len(found & truth)
    fp = len(found - truth)
    fn = len(truth - found)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    args = parser.parse_args()
    outputs_dir = Path(args.outputs_dir)

    rows = []
    for index, item in enumerate(load_questions(), 1):
        question_dir = str(item["directory"])
        spec = json.loads((ROOT / question_dir / "benchmark.json").read_text())
        truth = set(spec["ground_truth_movie_ids"])
        if not truth:
            raise RuntimeError(f"{question_dir} has empty ground truth")
        for metrics_path in sorted((outputs_dir / question_dir).glob("*/run_metrics.json")):
            run = json.loads(metrics_path.read_text())
            quality = quality_from_run(run, truth)
            rows.append(
                {
                    "question_index": index,
                    "question": question_dir,
                    "difficulty": spec["difficulty"],
                    "ground_truth_count": len(truth),
                    "implementation": run["implementation"],
                    "method": LABELS.get(run["implementation"], run["implementation"]),
                    "repetitions": run.get("repetitions", 1),
                    "wall_seconds": run["wall_seconds"],
                    "engine_seconds": run.get("engine_seconds", run["wall_seconds"]),
                    "cpu_seconds": run.get("cpu_seconds", 0.0),
                    "llm_calls": run["llm_calls"],
                    "cheap_calls": run.get("cheap_calls", 0),
                    "expensive_calls": run.get("expensive_calls", 0),
                    "cheap_seconds": run.get("cheap_seconds", 0.0),
                    "expensive_seconds": run.get("expensive_seconds", 0.0),
                    "final_answer_rows": run.get("final_answer_rows", 0),
                    **quality,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"No run_metrics.json files found under {outputs_dir}")
    frame["method"] = pd.Categorical(frame["method"], METHOD_ORDER, ordered=True)
    frame = frame.sort_values(["question_index", "method"])
    frame.to_csv(outputs_dir / "comparison.csv", index=False)

    aggregate = (
        frame.groupby(["implementation", "method"], as_index=False, observed=True)
        .agg(
            questions=("question", "count"),
            mean_wall_seconds=("wall_seconds", "mean"),
            total_wall_seconds=("wall_seconds", "sum"),
            mean_llm_calls=("llm_calls", "mean"),
            total_llm_calls=("llm_calls", "sum"),
            mean_cheap_calls=("cheap_calls", "mean"),
            mean_expensive_calls=("expensive_calls", "mean"),
            total_cheap_seconds=("cheap_seconds", "sum"),
            total_expensive_seconds=("expensive_seconds", "sum"),
            macro_precision=("precision", "mean"),
            macro_recall=("recall", "mean"),
            macro_f1=("f1", "mean"),
        )
        .sort_values("method")
    )
    model_seconds = aggregate["total_cheap_seconds"] + aggregate["total_expensive_seconds"]
    aggregate["cheap_time_percent"] = np.where(
        model_seconds > 0,
        100.0 * aggregate["total_cheap_seconds"] / model_seconds,
        0.0,
    )
    aggregate["expensive_time_percent"] = np.where(
        model_seconds > 0,
        100.0 * aggregate["total_expensive_seconds"] / model_seconds,
        0.0,
    )
    aggregate.to_csv(outputs_dir / "aggregate.csv", index=False)
    write_summary(outputs_dir, frame, aggregate)
    cleanup_old_plots(outputs_dir)
    plot_experiment_set(frame, outputs_dir)
    plot_question_sets(frame, outputs_dir)
    print(frame.to_string(index=False))
    print("\nAggregate:\n", aggregate.to_string(index=False))


def write_summary(outputs_dir: Path, frame: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    aggregate_columns = [
        "method",
        "questions",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "mean_wall_seconds",
        "mean_llm_calls",
    ]
    per_question_columns = [
        "question",
        "method",
        "repetitions",
        "precision",
        "recall",
        "f1",
        "wall_seconds",
        "llm_calls",
        "cheap_calls",
        "expensive_calls",
    ]
    lines = [
        "# Common Benchmark 10Q Summary",
        "",
        "Each question/method row reports the mean of the configured repetitions.",
        "",
        "## Aggregate",
        "",
        markdown_table(aggregate[aggregate_columns]),
        "",
        "## Per-question Metrics",
        "",
        markdown_table(frame[per_question_columns]),
        "",
    ]
    (outputs_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    rows = []
    columns = list(frame.columns)
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for record in frame.to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def cleanup_old_plots(outputs_dir: Path) -> None:
    for filename in [
        "metrics_precision_recall_f1.png",
        "quality_by_question.png",
        "time_bar_plot.png",
        "calls_bar_plot.png",
    ]:
        path = outputs_dir / filename
        if path.exists():
            path.unlink()
    legacy_question_dir = outputs_dir / "question_quality_plots"
    if legacy_question_dir.exists():
        shutil.rmtree(legacy_question_dir)


def plot_experiment_set(frame: pd.DataFrame, outputs_dir: Path) -> None:
    plots_dir = outputs_dir / "plots"
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    methods = [method for method in METHOD_ORDER if method in set(frame["method"].astype(str))]
    aggregate = (
        frame.groupby("method", as_index=False, observed=True)
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            wall_seconds=("wall_seconds", "mean"),
            llm_calls=("llm_calls", "mean"),
            cheap_calls=("cheap_calls", "mean"),
            expensive_calls=("expensive_calls", "mean"),
            cheap_seconds=("cheap_seconds", "mean"),
            expensive_seconds=("expensive_seconds", "mean"),
        )
        .set_index("method")
        .reindex(methods)
        .reset_index()
    )
    aggregate = aggregate[aggregate["method"].notna()].copy()
    note = "Whole experiment: each value is the mean across the 10 benchmark questions after each question/method was averaged over 11 repetitions."
    plot_question_quality(aggregate, plots_dir / "01_quality_precision_recall_f1.png", "Whole experiment", note)
    plot_question_time_split(aggregate, plots_dir / "02_time_cheap_expensive_percent.png", "Whole experiment", note)
    plot_question_call_split(aggregate, plots_dir / "03_calls_cheap_expensive_percent.png", "Whole experiment", note)
    plot_question_tradeoff(aggregate, plots_dir / "04_quality_time_calls_tradeoff.png", "Whole experiment", note)


def plot_question_sets(frame: pd.DataFrame, outputs_dir: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"].astype(str))]
    for question, question_frame in frame.groupby("question", sort=False, observed=True):
        question_dir = outputs_dir / str(question)
        plots_dir = question_dir / "plots"
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        ordered = (
            question_frame.set_index("method")
            .reindex(methods)
            .reset_index()
        )
        ordered = ordered[ordered["method"].notna()].copy()
        question_index = int(ordered["question_index"].iloc[0])
        question_text = read_question_text(str(question))
        title_prefix = f"Q{question_index}"
        plot_question_quality(ordered, plots_dir / "01_quality_precision_recall_f1.png", title_prefix, question_text)
        plot_question_time_split(ordered, plots_dir / "02_time_cheap_expensive_percent.png", title_prefix, question_text)
        plot_question_call_split(ordered, plots_dir / "03_calls_cheap_expensive_percent.png", title_prefix, question_text)
        plot_question_tradeoff(ordered, plots_dir / "04_quality_time_calls_tradeoff.png", title_prefix, question_text)


def plot_question_quality(frame: pd.DataFrame, path: Path, title_prefix: str, note_text: str) -> None:
    methods = list(frame["method"].astype(str))
    labels = [DISPLAY_LABELS[method] for method in methods]
    x = np.arange(len(methods))
    width = 0.22
    fig, ax = plt.subplots(figsize=(12, 6.8))
    for offset, metric_label, column in [
        (-width, "Precision", "precision"),
        (0.0, "Recall", "recall"),
        (width, "F1", "f1"),
    ]:
        values = frame[column].astype(float).tolist()
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=metric_label,
            color=QUALITY_COLORS[metric_label],
        )
        add_value_labels(ax, bars, "{:.3f}", y_offset=0.012)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title(f"{title_prefix}: precision, recall, and F1")
    ax.legend(loc="upper left", ncols=3, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    add_question_note(fig, note_text)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_question_time_split(frame: pd.DataFrame, path: Path, title_prefix: str, note_text: str) -> None:
    split_frame = frame.copy()
    cheap = split_frame["cheap_seconds"].astype(float).to_numpy()
    expensive = split_frame["expensive_seconds"].astype(float).to_numpy()
    wall = split_frame["wall_seconds"].astype(float).to_numpy()
    missing_split = cheap + expensive <= 0
    expensive = np.where(missing_split, wall, expensive)
    plot_percent_split(
        frame=split_frame,
        path=path,
        cheap=cheap,
        expensive=expensive,
        title_prefix=title_prefix,
        title="cheap vs expensive model time",
        ylabel="Model time (seconds)",
        cheap_label="Cheap model time",
        expensive_label="Expensive model time",
        value_unit="s",
        note_text=note_text,
    )


def plot_question_call_split(frame: pd.DataFrame, path: Path, title_prefix: str, note_text: str) -> None:
    plot_percent_split(
        frame=frame,
        path=path,
        cheap=frame["cheap_calls"].astype(float).to_numpy(),
        expensive=frame["expensive_calls"].astype(float).to_numpy(),
        title_prefix=title_prefix,
        title="cheap vs expensive model calls",
        ylabel="LLM calls",
        cheap_label="Cheap model calls",
        expensive_label="Expensive model calls",
        value_unit="calls",
        note_text=note_text,
    )


def plot_percent_split(
    frame: pd.DataFrame,
    path: Path,
    cheap: np.ndarray,
    expensive: np.ndarray,
    title_prefix: str,
    title: str,
    ylabel: str,
    cheap_label: str,
    expensive_label: str,
    value_unit: str,
    note_text: str,
) -> None:
    methods = list(frame["method"].astype(str))
    labels = [DISPLAY_LABELS[method] for method in methods]
    totals = cheap + expensive
    cheap_pct = np.divide(cheap, totals, out=np.zeros_like(cheap, dtype=float), where=totals > 0) * 100.0
    expensive_pct = np.divide(expensive, totals, out=np.zeros_like(expensive, dtype=float), where=totals > 0) * 100.0
    x = np.arange(len(methods))
    max_total = float(totals.max()) if len(totals) else 0.0
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.bar(x, cheap, width=0.62, color=CHEAP_COLOR, label=cheap_label)
    ax.bar(x, expensive, width=0.62, bottom=cheap, color=EXPENSIVE_COLOR, label=expensive_label)
    for index, (cheap_value, expensive_value, cheap_share, expensive_share, total) in enumerate(
        zip(cheap, expensive, cheap_pct, expensive_pct, totals)
    ):
        annotate_percent_segment(
            ax,
            index,
            cheap_value,
            0.0,
            cheap_share,
            max_total,
            value_unit,
            "cheap",
        )
        annotate_percent_segment(
            ax,
            index,
            expensive_value,
            cheap_value,
            expensive_share,
            max_total,
            value_unit,
            "expensive",
        )
        ax.text(
            index,
            total + y_padding(max_total),
            total_label(total, value_unit),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, stacked_ylim(totals))
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}: {title}")
    ax.legend(loc="upper left", ncols=2, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    add_question_note(fig, note_text)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_question_tradeoff(frame: pd.DataFrame, path: Path, title_prefix: str, note_text: str) -> None:
    methods = list(frame["method"].astype(str))
    wall = frame["wall_seconds"].astype(float).to_numpy()
    calls = frame["llm_calls"].astype(float).to_numpy()
    recall = frame["recall"].astype(float).to_numpy()
    best_index = int(np.argmin(tradeoff_distance(wall, calls, recall)))
    sizes = 220.0 + 42.0 * calls

    fig, ax = plt.subplots(figsize=(12, 7.2))
    for index, method in enumerate(methods):
        ax.scatter(
            wall[index],
            recall[index],
            s=sizes[index],
            color=COLORS[method],
            alpha=0.82,
            edgecolor="#222222" if index == best_index else "white",
            linewidth=2.2 if index == best_index else 1.0,
        )
        ax.text(
            wall[index],
            recall[index],
            f"#{index + 1}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        x_offset = -112 if wall[index] > np.percentile(wall, 75) else 8
        y_offset = -36 if recall[index] > np.percentile(recall, 75) else 8
        ax.annotate(
            f"{wall[index]:.2f}s, {calls[index]:.1f} calls\nRecall={recall[index]:.3f}",
            (wall[index], recall[index]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=8,
        )
    ax.scatter(
        wall[best_index],
        recall[best_index],
        s=sizes[best_index] + 260,
        facecolors="none",
        edgecolors="#111111",
        linewidths=2.4,
    )
    ax.set_xlabel("Mean wall time in seconds (lower is better)")
    ax.set_ylabel("Recall quality score (higher is better)")
    x_span = max(float(wall.max() - wall.min()), 1.0)
    y_span = max(float(recall.max() - recall.min()), 0.1)
    ax.set_xlim(float(wall.min()) - 0.08 * x_span, float(wall.max()) + 0.18 * x_span)
    ax.set_ylim(max(0.0, float(recall.min()) - 0.18 * y_span), min(1.05, float(recall.max()) + 0.25 * y_span))
    ax.set_title(
        f"{title_prefix}: quality/time/call tradeoff\n"
        f"Best normalized tradeoff: {SHORT_DISPLAY_LABELS[methods[best_index]]}"
    )
    ax.grid(alpha=0.25, linewidth=0.8)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[method],
            markeredgecolor="#222222" if index == best_index else "white",
            markeredgewidth=2.0 if index == best_index else 1.0,
            markersize=12,
            label=f"#{index + 1} {METHOD_CONTEXT[method]}",
        )
        for index, method in enumerate(methods)
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        title="Implementation",
        title_fontsize=9,
    )
    ax.text(
        0.02,
        0.04,
        "Top-left is better. Bubble size is mean LLM calls.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    add_question_note(fig, note_text)
    fig.tight_layout(rect=(0, 0.13, 0.84, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def tradeoff_distance(wall: np.ndarray, calls: np.ndarray, quality: np.ndarray) -> np.ndarray:
    wall_norm = normalize_low_is_good(wall)
    calls_norm = normalize_low_is_good(calls)
    quality_gap = normalize_high_is_good_gap(quality)
    return np.sqrt(wall_norm**2 + calls_norm**2 + quality_gap**2)


def normalize_low_is_good(values: np.ndarray) -> np.ndarray:
    span = float(values.max() - values.min())
    if span <= 0:
        return np.zeros_like(values, dtype=float)
    return (values - values.min()) / span


def normalize_high_is_good_gap(values: np.ndarray) -> np.ndarray:
    span = float(values.max() - values.min())
    if span <= 0:
        return np.zeros_like(values, dtype=float)
    return (values.max() - values) / span


def add_value_labels(ax, bars, fmt: str, y_offset: float) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90 if height > 0.85 else 0,
        )


def annotate_percent_segment(
    ax,
    x: int,
    value: float,
    bottom: float,
    share: float,
    max_total: float,
    value_unit: str,
    segment_name: str,
) -> None:
    if value <= 0 or share <= 0:
        return
    label = f"{share:.1f}%\n{value:.1f}{unit_suffix(value_unit)}"
    tall_enough = max_total > 0 and value / max_total >= 0.10
    inside = share >= 12 and tall_enough
    color = "white" if inside else "#222222"
    x_offset = 0.0 if inside else (0.34 if segment_name == "expensive" else -0.34)
    ha = "center" if inside else ("left" if segment_name == "expensive" else "right")
    ax.text(
        x + x_offset,
        bottom + value / 2,
        label,
        ha=ha,
        va="center",
        fontsize=8,
        color=color,
        rotation=90 if inside else 0,
        bbox=None if inside else {
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )


def total_label(total: float, value_unit: str) -> str:
    if value_unit == "s":
        return f"total {total:.1f}s"
    return f"total {total:.1f} calls"


def y_padding(max_total: float) -> float:
    return max(max_total * 0.025, 0.02)


def stacked_ylim(totals: np.ndarray) -> float:
    if len(totals) == 0:
        return 1.0
    max_total = float(np.max(totals))
    if max_total <= 0:
        return 1.0
    return max_total * 1.18


def unit_suffix(value_unit: str) -> str:
    if value_unit == "s":
        return "s"
    return ""


def add_question_note(fig, question_text: str) -> None:
    fig.text(
        0.02,
        0.035,
        textwrap.fill(question_text, width=120),
        ha="left",
        va="bottom",
        fontsize=9,
    )


def plot_quality(frame: pd.DataFrame, path: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"].astype(str))]
    aggregate = (
        frame.groupby("method", observed=True)[["precision", "recall", "f1"]]
        .mean()
        .reindex(methods)
    )
    x = np.arange(len(methods))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, aggregate["precision"], width, label="Precision", color=QUALITY_COLORS["Precision"])
    ax.bar(x, aggregate["recall"], width, label="Recall", color=QUALITY_COLORS["Recall"])
    ax.bar(x + width, aggregate["f1"], width, label="F1", color=QUALITY_COLORS["F1"])
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Aggregate quality metrics across 10 questions")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper left", ncols=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quality_by_question(frame: pd.DataFrame, path: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"].astype(str))]
    questions = list(frame.sort_values("question_index")["question"].drop_duplicates())
    positions, labels = grouped_positions(questions, methods)
    ordered_rows = ordered_question_method_rows(frame, questions, methods)
    fig, ax = plt.subplots(figsize=(22, 8))
    width = 0.22
    for offset, name, column in [
        (-width, "Precision", "precision"),
        (0.0, "Recall", "recall"),
        (width, "F1", "f1"),
    ]:
        ax.bar(
            [position + offset for position in positions],
            [row[column] for row in ordered_rows],
            width=width,
            label=name,
            color=QUALITY_COLORS[name],
        )
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Quality metrics by question and implementation")
    ax.legend(loc="upper left", ncols=3, frameon=False)
    common_axes(ax, questions, methods, labels, y=1.04, question_y=1.08)
    fig.tight_layout(rect=(0, 0, 0.69, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quality_per_question(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [method for method in METHOD_ORDER if method in set(frame["method"].astype(str))]
    metrics = [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]
    width = 0.22
    x = np.arange(len(methods))
    for _, question_frame in frame.groupby("question", sort=False, observed=True):
        question = str(question_frame["question"].iloc[0])
        question_index = int(question_frame["question_index"].iloc[0])
        ordered = (
            question_frame.set_index("method")
            .reindex(methods)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for offset, (label, metric) in zip([-width, 0.0, width], metrics):
            values = []
            for method in methods:
                row = ordered[ordered["method"].astype(str).eq(method)]
                value = row[metric].iloc[0] if not row.empty else 0.0
                values.append(float(value) if pd.notna(value) else 0.0)
            ax.bar(
                x + offset,
                values,
                width,
                label=label,
                color=QUALITY_COLORS[label],
            )
        ax.set_xticks(x, methods)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.set_title(f"Q{question_index}: quality metrics by implementation")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.legend(loc="upper left", ncols=3, frameon=False)
        question_text = read_question_text(question)
        ax.text(
            0.0,
            -0.21,
            "\n".join([textwrap.fill(question_text, width=92), method_context_text(methods)]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"q{question_index:02d}_{question}_quality.png", dpi=180)
        plt.close(fig)


def plot_metric(frame: pd.DataFrame, column: str, title: str, ylabel: str, path: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"].astype(str))]
    questions = list(frame.sort_values("question_index")["question"].drop_duplicates())
    positions, labels = grouped_positions(questions, methods)
    ordered = ordered_question_method_rows(frame, questions, methods)
    if column == "wall_seconds":
        cheap_col, expensive_col = "cheap_seconds", "expensive_seconds"
        cheap_label, expensive_label = "Cheap model time", "Expensive model time"
        annotation_mode = "percent"
    elif column == "llm_calls":
        cheap_col, expensive_col = "cheap_calls", "expensive_calls"
        cheap_label, expensive_label = "Cheap model calls", "Expensive model calls"
        annotation_mode = "percent"
    else:
        cheap_col, expensive_col = "", ""
        cheap_label, expensive_label = "", ""
        annotation_mode = "none"

    fig, ax = plt.subplots(figsize=(22, 8))
    if cheap_col and expensive_col:
        cheap = [float(row.get(cheap_col, 0.0)) for row in ordered]
        expensive = [float(row.get(expensive_col, 0.0)) for row in ordered]
        if column == "wall_seconds":
            expensive = [
                float(row[column]) if cheap_value + expensive_value <= 0 else expensive_value
                for row, cheap_value, expensive_value in zip(ordered, cheap, expensive)
            ]
        totals = [cheap_value + expensive_value for cheap_value, expensive_value in zip(cheap, expensive)]
        ax.bar(positions, cheap, width=0.62, color=CHEAP_COLOR, label=cheap_label)
        ax.bar(positions, expensive, width=0.62, bottom=cheap, color=EXPENSIVE_COLOR, label=expensive_label)
        for x_value, cheap_value, expensive_value in zip(positions, cheap, expensive):
            annotate_stack(ax, x_value, cheap_value, expensive_value, annotation_mode)
        ax.set_ylim(0, max(totals) * 1.22 if totals else 1.0)
    else:
        ax.bar(positions, [row[column] for row in ordered], width=0.62, color="#6B7280")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", ncols=2, frameon=False)
    common_axes(ax, questions, methods, labels, y=ax.get_ylim()[1] * 0.88, question_y=ax.get_ylim()[1] * 0.94)
    fig.tight_layout(rect=(0, 0, 0.69, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def grouped_positions(questions: list[str], methods: list[str]) -> tuple[list[int], list[str]]:
    positions = []
    labels = []
    for question_index, _ in enumerate(questions):
        for method_index, method in enumerate(methods):
            positions.append(question_index * len(methods) + method_index)
            labels.append(method)
    return positions, labels


def ordered_question_method_rows(frame: pd.DataFrame, questions: list[str], methods: list[str]) -> list[pd.Series]:
    rows = []
    for question in questions:
        subset = frame[frame["question"].astype(str).eq(question)]
        for method in methods:
            rows.append(subset[subset["method"].astype(str).eq(method)].iloc[0])
    return rows


def question_title(question_dir: str) -> str:
    parts = question_dir.split("_")
    if len(parts) >= 2 and parts[0] == "question":
        return f"Q{int(parts[1])}"
    return question_dir


def read_question_text(question_dir: str) -> str:
    spec = json.loads((ROOT / question_dir / "benchmark.json").read_text())
    return str(spec.get("question", question_dir))


def question_legend(questions: list[str]) -> str:
    lines = []
    for question in questions:
        lines.append(f"{question_title(question)}: {textwrap.fill(read_question_text(question), width=66)}")
    return "\n\n".join(lines)


def method_context_text(methods: list[str]) -> str:
    return "\n".join(METHOD_CONTEXT[method] for method in methods)


def add_question_boxes(ax, questions: list[str], methods: list[str], y: float) -> None:
    group = len(methods)
    for index, question in enumerate(questions):
        left = index * group - 0.5
        right = (index + 1) * group - 0.5
        center = index * group + (group - 1) / 2
        ax.axvline(left, color="#222222", linewidth=1.0)
        ax.text(center, y, question_title(question), ha="center", va="bottom", fontsize=10, fontweight="bold")
        if index == len(questions) - 1:
            ax.axvline(right, color="#222222", linewidth=1.0)


def common_axes(
    ax,
    questions: list[str],
    methods: list[str],
    labels: list[str],
    y: float,
    question_y: float,
) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    add_question_boxes(ax, questions, methods, question_y)
    ax.text(
        1.02,
        0.98,
        question_legend(questions) + "\n\n" + method_context_text(methods),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )


def annotate_stack(ax, x: int, cheap: float, expensive: float, mode: str) -> None:
    total = cheap + expensive
    if total <= 0:
        return
    if mode == "percent":
        cheap_text = f"{cheap / total * 100:.0f}% cheap"
        expensive_text = f"{expensive / total * 100:.0f}% expensive"
    else:
        cheap_text = f"{cheap:.1f} cheap"
        expensive_text = f"{expensive:.1f} expensive"
    if cheap > 0:
        ax.text(x, cheap / 2, cheap_text, ha="center", va="center", fontsize=6.5, color="white", rotation=90)
    if expensive > 0:
        ax.text(
            x,
            cheap + expensive / 2,
            expensive_text,
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
            rotation=90,
        )


if __name__ == "__main__":
    main()
