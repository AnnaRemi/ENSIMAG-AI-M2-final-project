#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_LABELS = {
    "suql_baseline": "SUQL",
    "trummer_heterogen_v1": "Trummer",
}
COLORS = {
    "SUQL": "red",
    "Trummer": "blue",
}
METRICS = [
    "cpu_seconds",
    "engine_seconds",
    "wall_seconds",
    "llm_calls",
    "final_answer_rows",
    "unique_found_movie_ids",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
]
TITLES = {
    "cpu_seconds": "Model vs CPU time",
    "engine_seconds": "Model vs engine time",
    "wall_seconds": "Model vs wall time",
    "llm_calls": "Model vs LLM calls",
    "final_answer_rows": "Model vs final answer rows",
    "unique_found_movie_ids": "Model vs unique found movie IDs",
    "true_positives": "Model vs true positives",
    "false_positives": "Model vs false positives",
    "false_negatives": "Model vs false negatives",
    "precision": "Model vs precision",
    "recall": "Model vs recall",
    "f1": "Model vs F1 score",
}
Y_LABELS = {
    "cpu_seconds": "CPU seconds",
    "engine_seconds": "Engine seconds",
    "wall_seconds": "Wall seconds",
    "llm_calls": "LLM calls",
    "final_answer_rows": "Rows",
    "unique_found_movie_ids": "Unique movie IDs",
    "true_positives": "True positives",
    "false_positives": "False positives",
    "false_negatives": "False negatives",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 score",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot every gathered metric across models for SUQL and Trummer."
    )
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--plot-dir",
        help="Destination directory. Defaults to <outputs-dir>/model_metric_plots.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    plot_dir = Path(args.plot_dir).resolve() if args.plot_dir else outputs_dir / "model_metric_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(outputs_dir.glob("*/comparison.csv")):
        frame = pd.read_csv(path)
        if set(frame["implementation"]) != set(IMPLEMENTATION_LABELS):
            print(f"Skipping incomplete comparison: {path}")
            continue
        frame["model_label"] = frame["model"].str.removeprefix("ollama/")
        frame["implementation_label"] = frame["implementation"].map(IMPLEMENTATION_LABELS)
        rows.append(frame)

    if not rows:
        raise SystemExit(f"No complete model comparison CSVs found under {outputs_dir}")

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values(["model_label", "implementation_label"]).reset_index(drop=True)
    combined.to_csv(plot_dir / "all_models_metrics.csv", index=False)

    model_order = list(dict.fromkeys(combined["model_label"]))
    for metric in METRICS:
        plot_metric(combined, model_order, metric, plot_dir / f"{metric}.png")

    print(f"Wrote {len(METRICS)} plots to {plot_dir}")
    print("Models:", ", ".join(model_order))


def plot_metric(frame: pd.DataFrame, model_order: list[str], metric: str, path: Path) -> None:
    pivot = frame.pivot(index="model_label", columns="implementation_label", values=metric)
    pivot = pivot.reindex(model_order)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = list(range(len(model_order)))
    for implementation in ("SUQL", "Trummer"):
        ax.plot(
            x,
            pivot[implementation],
            color=COLORS[implementation],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=implementation,
        )

    ax.set_xticks(x, model_order, rotation=25, ha="right")
    ax.set_xlabel("Ollama model")
    ax.set_ylabel(Y_LABELS[metric])
    ax.set_title(TITLES[metric])
    ax.grid(axis="both", alpha=0.25)
    ax.legend()
    if metric in {"precision", "recall", "f1"}:
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
