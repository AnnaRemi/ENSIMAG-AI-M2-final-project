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
    "trummer_heterogen_v2_2_structured_pruned": "V2_2",
    "trummer_heterogen_v2_3_batched_cascade": "V2_3",
    "trummer_heterogen_v3_pruned_cascade": "V3",
    "trummer_heterogen_v3_2_pruned_batched_cascade": "V3_2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    args = parser.parse_args()
    outputs_dir = Path(args.outputs_dir)
    rows = []
    for question_dir in ("question_1_easy", "question_2_medium", "question_3_hard"):
        spec = json.loads((ROOT / question_dir / "benchmark.json").read_text())
        truth = set(spec["ground_truth_movie_ids"])
        for metrics_path in sorted((outputs_dir / question_dir).glob("*/run_metrics.json")):
            run = json.loads(metrics_path.read_text())
            found = set(run["found_movie_ids"])
            tp = len(found & truth)
            fp = len(found - truth)
            fn = len(truth - found)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "question": question_dir,
                    "difficulty": spec["difficulty"],
                    "implementation": run["implementation"],
                    "method": LABELS.get(run["implementation"], run["implementation"]),
                    "wall_seconds": run["wall_seconds"],
                    "llm_calls": run["llm_calls"],
                    "cheap_calls": run.get("cheap_calls", 0),
                    "expensive_calls": run.get("expensive_calls", 0),
                    "cheap_seconds": run.get("cheap_seconds", 0.0),
                    "expensive_seconds": run.get("expensive_seconds", 0.0),
                    "cheap_time_percent": run.get("cheap_time_percent", 0.0),
                    "expensive_time_percent": run.get("expensive_time_percent", 0.0),
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["difficulty", "method"])
    if frame.empty:
        raise SystemExit(f"No run_metrics.json files found under {outputs_dir}")
    frame.to_csv(outputs_dir / "comparison.csv", index=False)
    aggregate = (
        frame.groupby(["implementation", "method"], as_index=False)
        .agg(
            total_wall_seconds=("wall_seconds", "sum"),
            total_llm_calls=("llm_calls", "sum"),
            total_cheap_calls=("cheap_calls", "sum"),
            total_expensive_calls=("expensive_calls", "sum"),
            total_cheap_seconds=("cheap_seconds", "sum"),
            total_expensive_seconds=("expensive_seconds", "sum"),
            macro_precision=("precision", "mean"),
            macro_recall=("recall", "mean"),
            macro_f1=("f1", "mean"),
        )
        .sort_values("method")
    )
    model_seconds = (
        aggregate["total_cheap_seconds"] + aggregate["total_expensive_seconds"]
    )
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
    plot(frame, aggregate, outputs_dir / "comparison.png")
    print(frame.to_string(index=False))
    print("\nAggregate:\n", aggregate.to_string(index=False))


def plot(frame: pd.DataFrame, aggregate: pd.DataFrame, path: Path) -> None:
    methods = [
        method
        for method in ["SUQL", "V2_2", "V2_3", "V3", "V3_2"]
        if method in set(frame["method"])
    ]
    questions = ["question_1_easy", "question_2_medium", "question_3_hard"]
    question_labels = ["Easy", "Medium", "Hard"]
    colors = {
        "SUQL": "#2878B5",
        "V2_2": "#E07A1F",
        "V2_3": "#8E5EA2",
        "V3": "#3A9D5D",
        "V3_2": "#4C9F70",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(questions))
    width = 0.19
    center = (len(methods) - 1) / 2
    for index, method in enumerate(methods):
        subset = (
            frame[frame["method"].eq(method)]
            .set_index("question")
            .reindex(questions)
        )
        axes[0, 0].bar(
            x + (index - center) * width,
            subset["f1"],
            width,
            label=method,
            color=colors[method],
        )
        axes[0, 1].bar(
            x + (index - center) * width,
            subset["wall_seconds"],
            width,
            color=colors[method],
        )
        axes[1, 0].bar(
            x + (index - center) * width,
            subset["llm_calls"],
            width,
            color=colors[method],
        )
    axes[0, 0].set_title("F1 by question difficulty")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend()
    axes[0, 1].set_title("Wall time by question")
    axes[0, 1].set_ylabel("Seconds")
    axes[1, 0].set_title("LLM calls by question")
    axes[1, 0].set_ylabel("Calls")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(x, question_labels)
        ax.grid(axis="y", alpha=0.25)

    ordered = aggregate.set_index("method").reindex(methods)
    agg_x = np.arange(len(methods))
    axes[1, 1].bar(
        agg_x - width / 2,
        ordered["macro_f1"],
        width,
        label="Macro F1",
        color="#6B7280",
    )
    max_time = max(float(ordered["total_wall_seconds"].max()), 1.0)
    axes[1, 1].bar(
        agg_x + width / 2,
        ordered["total_wall_seconds"] / max_time,
        width,
        label="Relative total time",
        color="#C44E52",
    )
    axes[1, 1].set_xticks(agg_x, methods)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Aggregate quality and relative runtime")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Three-question benchmark: SUQL vs Heterogen V2_2, V2_3, and V3",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
