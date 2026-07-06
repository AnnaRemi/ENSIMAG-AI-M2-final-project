#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = [
    "trummer_heterogen_v1",
    "trummer_heterogen_v2_cascade",
    "trummer_heterogen_v2_2_structured_pruned",
    "trummer_heterogen_v2_3_batched_cascade",
    "trummer_heterogen_v3_pruned_cascade",
]
LABELS = {
    "trummer_heterogen_v1": "V1",
    "trummer_heterogen_v2_cascade": "V2",
    "trummer_heterogen_v2_2_structured_pruned": "V2_2",
    "trummer_heterogen_v2_3_batched_cascade": "V2_3",
    "trummer_heterogen_v3_pruned_cascade": "V3",
}
FOCUS_IMPLEMENTATIONS = [
    "trummer_heterogen_v2_2_structured_pruned",
    "trummer_heterogen_v2_3_batched_cascade",
    "trummer_heterogen_v3_pruned_cascade",
]
FOCUS_LABELS = {
    "trummer_heterogen_v2_2_structured_pruned": "Structured Prunning\n& Block join\n(v2_2)",
    "trummer_heterogen_v2_3_batched_cascade": "Batch-wise cascading\n(v2_3)",
    "trummer_heterogen_v3_pruned_cascade": "Structured Prunning\n& Cascading\n(v3)",
}
COLORS = ["#2878B5", "#C44E52", "#55A868", "#E07A1F", "#8E5EA2", "#4C72B0"]
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
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Evaluate and plot available implementations instead of requiring all versions.",
    )
    args = parser.parse_args()
    outputs_dir = Path(args.outputs_dir)
    spec = json.loads((ROOT / "benchmark.json").read_text())
    truth = set(spec["ground_truth_movie_ids"])

    rows: list[dict] = []
    outcomes: list[dict] = []
    for path in sorted(outputs_dir.glob("*/run_metrics.json")):
        run = json.loads(path.read_text())
        implementation = run["implementation"]
        if implementation not in ORDER:
            continue
        found = set(run.get("found_movie_ids", []))
        tp, fp, fn, precision, recall, f1 = quality_from_run(run, found, truth)
        wall_seconds = float(run.get("wall_seconds", 0.0))
        input_movies = int(
            run.get("input_movies", run.get("original_movies", spec["row_count"]))
        )
        input_reviews = int(
            run.get("input_reviews", run.get("original_reviews", spec["row_count"]))
        )
        pruned_movies = int(run.get("pruned_movies", input_movies))
        pruned_reviews = int(run.get("pruned_reviews", input_reviews))
        candidate_pairs = int(
            run.get("candidate_pairs", pruned_movies * pruned_reviews)
        )
        row = {
            "implementation": implementation,
            "label": LABELS[implementation],
            "mode": run.get("mode", ""),
            "model": run.get("model", ""),
            "cheap_model": run.get("cheap_model", ""),
            "expensive_model": run.get("expensive_model", ""),
            "wall_seconds": wall_seconds,
            "engine_seconds": float(run.get("engine_seconds", wall_seconds)),
            "cpu_seconds": float(run.get("cpu_seconds", 0.0)),
            "llm_calls": float(run.get("llm_calls", 0)),
            "block_join_calls": float(run.get("block_join_calls", 0)),
            "cheap_calls": float(run.get("cheap_calls", 0)),
            "expensive_calls": float(run.get("expensive_calls", 0)),
            "planned_cheap_calls": float(
                run.get("planned_cheap_calls", run.get("cheap_calls", 0))
            ),
            "planned_expensive_calls": float(
                run.get("planned_expensive_calls", run.get("expensive_calls", 0))
            ),
            "cheap_seconds": float(run.get("cheap_seconds", 0.0)),
            "expensive_seconds": float(run.get("expensive_seconds", 0.0)),
            "cheap_time_percent": float(run.get("cheap_time_percent", 0.0)),
            "expensive_time_percent": float(
                run.get("expensive_time_percent", 0.0)
            ),
            "cheap_early_accepts": float(run.get("cheap_early_accepts", 0)),
            "cheap_early_rejects": float(run.get("cheap_early_rejects", 0)),
            "cheap_failures": float(run.get("cheap_failures", 0)),
            "cheap_failure_candidates": float(
                run.get("cheap_failure_candidates", 0)
            ),
            "expensive_candidates": float(run.get("expensive_candidates", 0)),
            "expensive_accepts": float(run.get("expensive_accepts", 0)),
            "expensive_failures": float(run.get("expensive_failures", 0)),
            "input_movies": input_movies,
            "input_reviews": input_reviews,
            "pruned_movies": pruned_movies,
            "pruned_reviews": pruned_reviews,
            "candidate_pairs": float(candidate_pairs),
            "final_answer_rows": float(run.get("final_answer_rows", len(found))),
            "prompt_tokens": float(run.get("prompt_tokens", 0)),
            "completion_tokens": float(run.get("completion_tokens", 0)),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "candidates_per_wall_second": (
                candidate_pairs / wall_seconds
                if wall_seconds > 0
                else 0.0
            ),
        }
        rows.append(row)
        for movie_id in sorted(truth | found):
            outcomes.append(
                {
                    "implementation": implementation,
                    "label": LABELS[implementation],
                    "movie_id": movie_id,
                    "ground_truth": int(movie_id in truth),
                    "found": int(movie_id in found),
                    "classification": (
                        "TP"
                        if movie_id in truth and movie_id in found
                        else "FP"
                        if movie_id in found
                        else "FN"
                    ),
                }
            )

    found_implementations = {row["implementation"] for row in rows}
    missing = [item for item in ORDER if item not in found_implementations]
    if missing and not args.allow_missing:
        raise SystemExit(
            "Missing run_metrics.json for: " + ", ".join(LABELS[item] for item in missing)
        )
    if not rows:
        raise SystemExit(f"No run_metrics.json files found under {outputs_dir}")

    present_order = [item for item in ORDER if item in found_implementations]
    frame = pd.DataFrame(rows).set_index("implementation").loc[present_order].reset_index()
    v1_rows = frame.loc[
        frame["implementation"].eq("trummer_heterogen_v1"),
        "wall_seconds",
    ]
    if v1_rows.empty:
        frame["speedup_vs_v1"] = 0.0
    else:
        v1_wall = float(v1_rows.iloc[0])
        frame["speedup_vs_v1"] = np.where(
            frame["wall_seconds"] > 0,
            v1_wall / frame["wall_seconds"],
            0.0,
        )
    frame.to_csv(outputs_dir / "all_metrics.csv", index=False)
    pd.DataFrame(outcomes).to_csv(
        outputs_dir / "movie_id_outcomes.csv",
        index=False,
    )
    write_summary(frame, spec, outputs_dir / "summary.md")
    plot_requested(frame, outputs_dir, allow_missing=args.allow_missing)
    print(frame.to_string(index=False), flush=True)


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


def write_summary(frame: pd.DataFrame, spec: dict, path: Path) -> None:
    columns = [
        "label",
        "wall_seconds",
        "llm_calls",
        "cheap_calls",
        "expensive_calls",
        "cheap_seconds",
        "expensive_seconds",
        "cheap_time_percent",
        "expensive_time_percent",
        "precision",
        "recall",
        "f1",
    ]
    lines = [
        "# All Heterogen versions: one-question comparison",
        "",
        f"Question: {spec['question']}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
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
        "Stage percentages use cheap plus expensive model-call time as the denominator.",
        "For V1 and V2_2 all LLM work is classified as expensive.",
        "Token metrics are available for block-join implementations; unavailable metrics remain zero.",
        "",
        "Generated plots:",
        "",
        "- `metrics_precision_recall_f1.png`",
        "- `time_bar_plot.png`",
        "- `calls_bar_plot.png`",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_requested(frame: pd.DataFrame, outputs_dir: Path, allow_missing: bool) -> None:
    focus = focus_frame(frame, allow_missing)
    clean_plot_outputs(outputs_dir)
    plot_focus_metrics(focus, outputs_dir / "metrics_precision_recall_f1.png")
    plot_focus_time(focus, outputs_dir / "time_bar_plot.png")
    plot_focus_calls(focus, outputs_dir / "calls_bar_plot.png")


def focus_frame(frame: pd.DataFrame, allow_missing: bool) -> pd.DataFrame:
    indexed = frame.set_index("implementation")
    missing = [item for item in FOCUS_IMPLEMENTATIONS if item not in indexed.index]
    if missing and not allow_missing:
        raise SystemExit(
            "Missing run_metrics.json for focused plots: "
            + ", ".join(FOCUS_LABELS[item] for item in missing)
        )
    present = [item for item in FOCUS_IMPLEMENTATIONS if item in indexed.index]
    if not present:
        raise SystemExit("No V2_2, V2_3, or V3 metrics found for plotting")
    focus = indexed.loc[present].copy()
    focus["plot_label"] = [FOCUS_LABELS[item] for item in present]
    return focus


def clean_plot_outputs(outputs_dir: Path) -> None:
    for path in outputs_dir.glob("*.png"):
        if path.name not in PLOT_FILES:
            path.unlink()


def plot_focus_metrics(frame: pd.DataFrame, path: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    plot_labels = frame["plot_label"].tolist()
    x = np.arange(len(frame))
    width = 0.23
    fig, ax = plt.subplots(figsize=(11, 6.2))
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
    ax.set_xticks(x, plot_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Precision, recall, and F1: V2_2 vs V2_3 vs V3")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_focus_time(frame: pd.DataFrame, path: Path) -> None:
    plot_labels = frame["plot_label"].tolist()
    wall = frame["wall_seconds"].astype(float).to_numpy()
    cheap_seconds = frame["cheap_seconds"].astype(float).clip(lower=0).to_numpy()
    expensive_seconds = frame["expensive_seconds"].astype(float).clip(lower=0).to_numpy()
    cheap_part, expensive_part, cheap_pct, expensive_pct = split_wall_time(
        wall,
        cheap_seconds,
        expensive_seconds,
    )
    fig, ax = plt.subplots(figsize=(11, 6.2))
    cheap_bars = ax.bar(
        plot_labels,
        cheap_part,
        color=CHEAP_COLOR,
        label="Cheap-model time",
    )
    expensive_bars = ax.bar(
        plot_labels,
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
    ax.set_title("Wall time split by cheap and expensive model-call time: V2_2 vs V2_3 vs V3")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
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
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{percent:.0f}%\n{label}",
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            fontsize=9,
        )


def plot_focus_calls(frame: pd.DataFrame, path: Path) -> None:
    plot_labels = frame["plot_label"].tolist()
    cheap = frame["cheap_calls"].astype(float).to_numpy()
    expensive = frame["expensive_calls"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    cheap_bars = ax.bar(
        plot_labels,
        cheap,
        color=CHEAP_COLOR,
        label="Cheap-model calls",
    )
    expensive_bars = ax.bar(
        plot_labels,
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
    ax.set_title("LLM calls split by cheap and expensive model calls: V2_2 vs V2_3 vs V3")
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


def labels(frame: pd.DataFrame) -> list[str]:
    return frame["label"].tolist()


def format_count(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quality(frame: pd.DataFrame, path: Path) -> None:
    ax = frame.set_index("label")[["precision", "recall", "f1"]].plot(
        kind="bar",
        figsize=(10, 5),
        rot=0,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval quality")
    ax.grid(axis="y", alpha=0.25)
    finish(ax.figure, path)


def plot_runtime(frame: pd.DataFrame, path: Path) -> None:
    ax = frame.set_index("label")[
        ["wall_seconds", "engine_seconds", "cpu_seconds"]
    ].plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime metrics")
    ax.grid(axis="y", alpha=0.25)
    finish(ax.figure, path)


def plot_calls(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels(frame), frame["cheap_calls"], label="cheap calls")
    ax.bar(
        labels(frame),
        frame["expensive_calls"],
        bottom=frame["cheap_calls"],
        label="expensive calls",
    )
    ax.plot(
        labels(frame),
        frame["llm_calls"],
        color="black",
        marker="o",
        label="total calls",
    )
    ax.set_title("LLM calls by stage")
    ax.set_ylabel("Calls")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_stage_seconds(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels(frame), frame["cheap_seconds"], label="cheap time")
    ax.bar(
        labels(frame),
        frame["expensive_seconds"],
        bottom=frame["cheap_seconds"],
        label="expensive time",
    )
    ax.set_title("Aggregate model-call time by stage")
    ax.set_ylabel("Seconds")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_stage_percent(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels(frame), frame["cheap_time_percent"], label="cheap %")
    ax.bar(
        labels(frame),
        frame["expensive_time_percent"],
        bottom=frame["cheap_time_percent"],
        label="expensive %",
    )
    ax.set_ylim(0, 105)
    ax.set_title("Model-call time percentage by stage")
    ax.set_ylabel("Percent")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    finish(fig, path)


def plot_routing(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "cheap_early_accepts",
        "cheap_early_rejects",
        "expensive_candidates",
        "cheap_failures",
        "expensive_failures",
    ]
    ax = frame.set_index("label")[columns].plot(
        kind="bar",
        figsize=(11, 5),
        rot=0,
    )
    ax.set_title("Cascade routing and failures")
    ax.set_ylabel("Candidates or calls")
    ax.grid(axis="y", alpha=0.25)
    finish(ax.figure, path)


def plot_workload(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "input_movies",
        "pruned_movies",
        "candidate_pairs",
        "final_answer_rows",
    ]
    ax = frame.set_index("label")[columns].plot(
        kind="bar",
        figsize=(11, 5),
        rot=0,
        logy=True,
    )
    ax.set_title("Workload size (log scale)")
    ax.set_ylabel("Rows or candidate pairs")
    ax.grid(axis="y", alpha=0.25)
    finish(ax.figure, path)


def plot_tokens(frame: pd.DataFrame, path: Path) -> None:
    ax = frame.set_index("label")[["prompt_tokens", "completion_tokens"]].plot(
        kind="bar",
        figsize=(10, 5),
        rot=0,
    )
    ax.set_title("Recorded token usage")
    ax.set_ylabel("Tokens")
    ax.grid(axis="y", alpha=0.25)
    finish(ax.figure, path)


def plot_confusion(frame: pd.DataFrame, path: Path) -> None:
    ax = frame.set_index("label")[
        ["true_positives", "false_positives", "false_negatives"]
    ].plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_title("Retrieval outcomes")
    ax.set_ylabel("Movies")
    ax.grid(axis="y", alpha=0.25)
    finish(ax.figure, path)


def plot_normalized_heatmap(frame: pd.DataFrame, path: Path) -> None:
    columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    values = frame[columns].astype(float)
    maxima = values.max(axis=0).replace(0, 1)
    normalized = values / maxima
    fig, ax = plt.subplots(figsize=(max(16, len(columns) * 0.55), 6))
    image = ax.imshow(normalized.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(frame)), labels(frame))
    ax.set_xticks(range(len(columns)), columns, rotation=55, ha="right")
    ax.set_title("All numeric metrics normalized by column maximum")
    fig.colorbar(image, ax=ax, label="Normalized value")
    finish(fig, path)


def plot_dashboard(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    x = np.arange(len(frame))
    names = labels(frame)
    axes[0, 0].bar(names, frame["wall_seconds"], color=COLORS)
    axes[0, 0].set_title("Wall time")
    axes[0, 1].bar(names, frame["cheap_calls"], label="cheap")
    axes[0, 1].bar(
        names,
        frame["expensive_calls"],
        bottom=frame["cheap_calls"],
        label="expensive",
    )
    axes[0, 1].set_title("LLM calls")
    axes[0, 1].legend()
    width = 0.25
    for offset, metric in enumerate(["precision", "recall", "f1"]):
        axes[0, 2].bar(
            x + (offset - 1) * width,
            frame[metric],
            width,
            label=metric,
        )
    axes[0, 2].set_xticks(x, names)
    axes[0, 2].set_ylim(0, 1.05)
    axes[0, 2].set_title("Quality")
    axes[0, 2].legend()
    axes[1, 0].bar(names, frame["cheap_seconds"], label="cheap")
    axes[1, 0].bar(
        names,
        frame["expensive_seconds"],
        bottom=frame["cheap_seconds"],
        label="expensive",
    )
    axes[1, 0].set_title("Model-call seconds")
    axes[1, 0].legend()
    axes[1, 1].bar(names, frame["cheap_time_percent"], label="cheap %")
    axes[1, 1].bar(
        names,
        frame["expensive_time_percent"],
        bottom=frame["cheap_time_percent"],
        label="expensive %",
    )
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_title("Stage-time percentage")
    axes[1, 1].legend()
    axes[1, 2].bar(names, frame["speedup_vs_v1"], color=COLORS)
    axes[1, 2].axhline(1.0, color="black", linewidth=1)
    axes[1, 2].set_title("Wall-time speedup vs V1")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("All Heterogen versions on one fixed question", fontsize=16)
    finish(fig, path)


if __name__ == "__main__":
    main()
