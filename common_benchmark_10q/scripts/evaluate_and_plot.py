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

from common import ROOT


LABELS = {
    "suql_baseline": "SUQL",
    "trummer_heterogen_v2_3_batched_cascade": "V2_3",
    "trummer_heterogen_v3_pruned_cascade": "V3",
    "trummer_heterogen_v3_2_pruned_batched_cascade": "V3_2",
}
METHOD_ORDER = ["SUQL", "V2_3", "V3", "V3_2"]
COLORS = {
    "SUQL": "#2878B5",
    "V2_3": "#8E5EA2",
    "V3": "#3A9D5D",
    "V3_2": "#4C9F70",
}


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
    plot_quality(frame, outputs_dir / "metrics_precision_recall_f1.png")
    plot_quality_by_question(frame, outputs_dir / "quality_by_question.png")
    plot_quality_per_question(frame, outputs_dir / "question_quality_plots")
    plot_metric(frame, "wall_seconds", "Mean wall time by question", "Seconds", outputs_dir / "time_bar_plot.png")
    plot_metric(frame, "llm_calls", "Mean LLM calls by question", "Calls", outputs_dir / "calls_bar_plot.png")
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


def plot_quality(frame: pd.DataFrame, path: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"].astype(str))]
    aggregate = (
        frame.groupby("method", observed=True)[["precision", "recall", "f1"]]
        .mean()
        .reindex(methods)
    )
    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, aggregate["precision"], width, label="Precision", color="#2878B5")
    ax.bar(x, aggregate["recall"], width, label="Recall", color="#E07A1F")
    ax.bar(x + width, aggregate["f1"], width, label="F1", color="#3A9D5D")
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean quality metrics across 10 questions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quality_by_question(frame: pd.DataFrame, path: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"].astype(str))]
    questions = list(frame.sort_values("question_index")["question"].drop_duplicates())
    labels = [f"Q{index}" for index in range(1, len(questions) + 1)]
    x = np.arange(len(questions))
    width = min(0.18, 0.75 / max(len(methods), 1))
    center = (len(methods) - 1) / 2
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for ax, column, title in zip(
        axes,
        ["precision", "recall", "f1"],
        ["Precision by question", "Recall by question", "F1 by question"],
    ):
        for index, method in enumerate(methods):
            subset = (
                frame[frame["method"].astype(str).eq(method)]
                .set_index("question")
                .reindex(questions)
            )
            ax.bar(
                x + (index - center) * width,
                subset[column],
                width,
                label=method,
                color=COLORS[method],
            )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(column.title())
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, labels)
    axes[0].legend(ncol=len(methods))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quality_per_question(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [method for method in METHOD_ORDER if method in set(frame["method"].astype(str))]
    metrics = ["precision", "recall", "f1"]
    width = 0.24
    x = np.arange(len(metrics))
    for _, question_frame in frame.groupby("question", sort=False, observed=True):
        question = str(question_frame["question"].iloc[0])
        question_index = int(question_frame["question_index"].iloc[0])
        ordered = (
            question_frame.set_index("method")
            .reindex(methods)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        center = (len(methods) - 1) / 2
        for index, method in enumerate(methods):
            row = ordered[ordered["method"].astype(str).eq(method)]
            values = []
            for metric in metrics:
                value = row[metric].iloc[0] if not row.empty else 0.0
                values.append(float(value) if pd.notna(value) else 0.0)
            ax.bar(
                x + (index - center) * width,
                values,
                width,
                label=method,
                color=COLORS[method],
            )
        ax.set_xticks(x, ["Precision", "Recall", "F1"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"Q{question_index}: quality metrics")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"q{question_index:02d}_{question}_quality.png", dpi=180)
        plt.close(fig)


def plot_metric(frame: pd.DataFrame, column: str, title: str, ylabel: str, path: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"].astype(str))]
    questions = list(frame.sort_values("question_index")["question"].drop_duplicates())
    labels = [f"Q{index}" for index in range(1, len(questions) + 1)]
    x = np.arange(len(questions))
    width = min(0.18, 0.75 / max(len(methods), 1))
    center = (len(methods) - 1) / 2
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for index, method in enumerate(methods):
        subset = frame[frame["method"].astype(str).eq(method)].set_index("question").reindex(questions)
        ax.bar(
            x + (index - center) * width,
            subset[column],
            width,
            label=method,
            color=COLORS[method],
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=len(methods))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
