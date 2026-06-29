#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate common benchmark runs and create reports.")
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    benchmark = json.loads((ROOT / "benchmark.json").read_text())
    truth = set(benchmark["ground_truth_movie_ids"])
    metric_files = sorted(outputs_dir.glob("*/run_metrics.json"))
    if not metric_files:
        raise SystemExit(f"No run_metrics.json files found under {outputs_dir}")

    rows = []
    found_rows = []
    for path in metric_files:
        run = json.loads(path.read_text())
        found = set(run["found_movie_ids"])
        tp, fp, fn, precision, recall, f1 = quality_from_run(run, found, truth)
        rows.append(
            {
                "implementation": run["implementation"],
                "mode": run["mode"],
                "model": run["model"],
                "cpu_seconds": run["cpu_seconds"],
                "engine_seconds": run["engine_seconds"],
                "wall_seconds": run["wall_seconds"],
                "llm_calls": run["llm_calls"],
                "final_answer_rows": run["final_answer_rows"],
                "unique_found_movie_ids": len(found),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        for movie_id in sorted(truth | found):
            found_rows.append(
                {
                    "implementation": run["implementation"],
                    "movie_id": movie_id,
                    "ground_truth": int(movie_id in truth),
                    "found": int(movie_id in found),
                    "classification": (
                        "TP" if movie_id in truth and movie_id in found
                        else "FP" if movie_id in found
                        else "FN"
                    ),
                }
            )

    comparison = pd.DataFrame(rows).sort_values("implementation").reset_index(drop=True)
    comparison.to_csv(outputs_dir / "comparison.csv", index=False)
    pd.DataFrame(found_rows).to_csv(outputs_dir / "movie_id_outcomes.csv", index=False)
    write_markdown(comparison, outputs_dir / "comparison.md", benchmark["question"])
    plot_times(comparison, outputs_dir / "time_comparison.png")
    plot_workload_quality(comparison, outputs_dir / "workload_quality_comparison.png")
    print(comparison.to_string(index=False))


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


def write_markdown(df: pd.DataFrame, path: Path, question: str) -> None:
    display = df[
        [
            "implementation",
            "mode",
            "cpu_seconds",
            "engine_seconds",
            "llm_calls",
            "final_answer_rows",
            "precision",
            "recall",
            "f1",
        ]
    ].copy()
    for column in ["cpu_seconds", "engine_seconds", "precision", "recall", "f1"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    headers = list(display.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in display.itertuples(index=False, name=None):
        markdown_rows.append("| " + " | ".join(map(str, values)) + " |")
    lines = [
        "# Common benchmark comparison",
        "",
        f"Question: {question}",
        "",
        *markdown_rows,
        "",
        "Precision and recall use unique movie IDs. `final_answer_rows` retains each implementation's raw output-row count.",
        "CPU time is the benchmark client process CPU time; external Ollama server CPU/GPU time is not included.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_times(df: pd.DataFrame, path: Path) -> None:
    ax = df.set_index("implementation")[["cpu_seconds", "engine_seconds"]].plot(
        kind="bar", figsize=(9, 5), rot=0
    )
    ax.set_ylabel("Seconds")
    ax.set_title("Common benchmark timing")
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=180)
    plt.close(ax.figure)


def plot_workload_quality(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    names = df["implementation"]
    axes[0].bar(names, df["llm_calls"])
    axes[0].set_title("LLM calls")
    axes[0].set_ylabel("Count")
    axes[1].bar(names, df["final_answer_rows"])
    axes[1].set_title("Final answer rows")
    axes[1].set_ylabel("Count")
    quality = df.set_index("implementation")[["precision", "recall", "f1"]]
    quality.plot(kind="bar", ax=axes[2], rot=0)
    axes[2].set_title("Unique-ID retrieval quality")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Score")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
