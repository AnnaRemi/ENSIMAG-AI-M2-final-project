#!/usr/bin/env python3
"""Configurable plotting for SUQL benchmark metrics.

This is the single plotting entry point for the SUQL project. It can plot:

- per-question metrics from one benchmark `metrics.csv`;
- scaling metrics from many benchmark `metrics.csv` files.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#7c3aed",
    "#f97316",
    "#0f766e",
    "#be123c",
]

DEFAULT_METRICS = [
    "wall_seconds",
    "engine_seconds",
    "llm_prompts_issued",
    "structured_candidates",
    "semantic_rows",
    "join_rows",
    "result_rows",
]

METRIC_LABELS = {
    "wall_seconds": ("Wall-clock Latency", "seconds"),
    "engine_seconds": ("Engine Time", "seconds"),
    "llm_full_calls": ("Full LLM Calls", "count"),
    "llm_prompts_issued": ("LLM Prompts Issued", "count"),
    "structured_candidates": ("Structured Candidates", "count"),
    "semantic_rows": ("Semantic Rows Retrieved", "count"),
    "join_rows": ("Join Rows", "count"),
    "result_rows": ("Result Rows", "count"),
    "cheap_score_calls": ("Cheap Score Calls", "count"),
    "cheap_score_failures": ("Cheap Score Failures", "count"),
    "expensive_full_calls": ("Expensive Full Calls", "count"),
    "cheap_early_accept": ("Cheap Early Accepts", "count"),
    "cheap_early_reject": ("Cheap Early Rejects", "count"),
    "cheap_skipped": ("Cheap Skipped", "count"),
    "cheap_disabled": ("Cheap Disabled", "count"),
}

FALLBACK_COLUMNS = {
    "engine_seconds": "wall_seconds",
    "llm_prompts_issued": "llm_full_calls",
    "structured_candidates": "result_rows",
    "semantic_rows": "result_rows",
    "join_rows": "result_rows",
    "expensive_full_calls": "llm_full_calls",
}


def parse_list(value: str | list[str] | None, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        values = value
    else:
        values = value.split(",") if "," in value else value.split()
    return [item.strip() for item in values if item.strip()]


def ensure_metric_columns(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    df = df.copy()
    for metric in metrics:
        fallback = FALLBACK_COLUMNS.get(metric)
        if metric not in df and fallback in df:
            df[metric] = df[fallback]
        if metric not in df:
            df[metric] = np.nan
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def color_map(implementations: list[str]) -> dict[str, str]:
    return {
        implementation: DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
        for index, implementation in enumerate(implementations)
    }


def metric_title(metric: str) -> tuple[str, str]:
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    return metric.replace("_", " ").title(), "value"


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#e5e7eb", linewidth=0.9, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_per_question_metrics(
    metrics_csv: Path,
    output: Path,
    implementations: list[str] | None = None,
    metrics: list[str] | None = None,
    title: str | None = None,
) -> None:
    df = pd.read_csv(metrics_csv)
    if "project" not in df or "query_id" not in df:
        raise ValueError(f"{metrics_csv} must include project and query_id columns")

    metrics = metrics or DEFAULT_METRICS
    df = ensure_metric_columns(df, metrics)
    implementations = implementations or list(dict.fromkeys(df["project"].dropna().astype(str)))
    question_rows = df.drop_duplicates("query_id")
    query_ids = list(question_rows["query_id"].astype(str))
    question_by_id = dict(zip(question_rows["query_id"].astype(str), question_rows.get("question", question_rows["query_id"]).astype(str)))
    labels = [f"Q{i + 1}" for i in range(len(query_ids))]
    x = np.arange(len(query_ids))
    colors = color_map(implementations)

    cols = 2
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, max(6, rows * 5.4)))
    axes_flat = np.atleast_1d(axes).flatten()
    fig.subplots_adjust(top=0.78 if query_ids else 0.90, hspace=0.48, wspace=0.24)

    if query_ids:
        question_lines = ["Question Index"]
        question_lines.extend(f"Q{i + 1} - {question_by_id[query_id]}" for i, query_id in enumerate(query_ids))
        fig.text(
            0.5,
            0.985,
            "\n".join(question_lines),
            ha="center",
            va="top",
            fontsize=9,
            fontfamily="monospace",
            fontweight="bold",
            linespacing=1.2,
        )

    for ax, metric in zip(axes_flat, metrics):
        metric_label, ylabel = metric_title(metric)
        for implementation in implementations:
            impl_df = df[df["project"].astype(str) == implementation].set_index("query_id").reindex(query_ids)
            values = impl_df[metric]
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=2.4,
                markersize=6,
                color=colors[implementation],
                label=implementation,
            )
            ax.fill_between(x, values.fillna(0), color=colors[implementation], alpha=0.08)
            for xi, value in zip(x, values):
                if pd.notna(value):
                    ax.annotate(
                        f"{value:.0f}",
                        (xi, value),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        color=colors[implementation],
                        fontsize=8,
                        fontweight="bold",
                    )

        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ax.legend(fontsize=8, loc="best")

    for ax in axes_flat[len(metrics):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, y=0.80 if query_ids else 0.98, fontsize=15, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sample_size_from_path(path: Path, regex: str) -> int:
    match = re.search(regex, str(path))
    if not match:
        raise ValueError(f"Cannot extract sample size from {path} with regex {regex!r}")
    return int(match.group(1))


def load_scaling_metrics(paths: list[Path], sample_regex: str) -> pd.DataFrame:
    frames = []
    for metrics_path in paths:
        frame = pd.read_csv(metrics_path)
        frame["sample_size"] = sample_size_from_path(metrics_path, sample_regex)
        frame["experiment"] = metrics_path.parent.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No metrics.csv files matched the scaling input")
    return pd.concat(frames, ignore_index=True)


def aggregate_scaling(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    data = ensure_metric_columns(data, metrics)
    data["exit_code"] = pd.to_numeric(data.get("exit_code", 0), errors="coerce").fillna(-1).astype(int)
    ok_counts = (
        data.groupby(["sample_size", "project"], as_index=False)
        .agg(
            ok_queries=("exit_code", lambda values: int((values == 0).sum())),
            failed_queries=("exit_code", lambda values: int((values != 0).sum())),
        )
    )
    successful = data[data["exit_code"] == 0].copy()
    summary = (
        successful.groupby(["sample_size", "project"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .merge(ok_counts, on=["sample_size", "project"], how="right")
        .fillna(0)
        .sort_values(["sample_size", "project"])
    )
    return summary


def plot_scaling_metrics(
    summary: pd.DataFrame,
    output: Path,
    implementations: list[str] | None = None,
    metrics: list[str] | None = None,
    title: str | None = None,
) -> None:
    metrics = metrics or DEFAULT_METRICS
    implementations = implementations or list(dict.fromkeys(summary["project"].dropna().astype(str)))
    colors = color_map(implementations)
    cols = 2
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(17, max(5.5, rows * 5.2)))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, metric in zip(axes_flat, metrics):
        metric_label, ylabel = metric_title(metric)
        for implementation in implementations:
            impl_df = summary[summary["project"].astype(str) == implementation].sort_values("sample_size")
            ax.plot(
                impl_df["sample_size"],
                impl_df[metric],
                marker="o",
                linewidth=2.4,
                color=colors[implementation],
                label=implementation,
            )
        ax.set_title(metric_label, fontweight="bold")
        ax.set_xlabel("sample size")
        ax.set_ylabel(f"mean {ylabel}")
        ax.legend()
        style_axes(ax)

    for ax in axes_flat[len(metrics):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def command_per_question(args: argparse.Namespace) -> None:
    plot_per_question_metrics(
        metrics_csv=args.metrics_csv,
        output=args.output,
        implementations=parse_list(args.impls),
        metrics=parse_list(args.metrics, DEFAULT_METRICS),
        title=args.title,
    )
    print(f"Wrote {args.output}")


def command_scaling(args: argparse.Namespace) -> None:
    if args.metrics_csv:
        paths = [Path(path) for path in args.metrics_csv]
    else:
        paths = sorted(args.benchmark_dir.glob(args.pattern))
    metrics = parse_list(args.metrics, DEFAULT_METRICS)
    data = load_scaling_metrics(paths, args.sample_regex)
    summary = aggregate_scaling(data, metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "detailed_metrics.csv", data)
    plot_scaling_metrics(
        summary=summary,
        output=args.output_dir / args.output_name,
        implementations=parse_list(args.impls),
        metrics=metrics,
        title=args.title,
    )
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'detailed_metrics.csv'}")
    print(f"Wrote {args.output_dir / args.output_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot SUQL benchmark metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    per_question = subparsers.add_parser("per-question", help="Plot metrics by benchmark question.")
    per_question.add_argument("--metrics-csv", type=Path, required=True)
    per_question.add_argument("--output", type=Path, required=True)
    per_question.add_argument("--impls", default="", help="Comma- or space-separated implementations to plot.")
    per_question.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    per_question.add_argument("--title", default="")
    per_question.set_defaults(func=command_per_question)

    scaling = subparsers.add_parser("scaling", help="Plot metrics over sample sizes.")
    scaling.add_argument("--benchmark-dir", type=Path, default=ROOT / "Stage_1" / "benchmarks")
    scaling.add_argument("--pattern", default="aker_baseline_stage1_data_sample_*/metrics.csv")
    scaling.add_argument("--metrics-csv", nargs="*", help="Explicit metrics.csv files to aggregate.")
    scaling.add_argument("--sample-regex", default=r"data_sample_(\d+)")
    scaling.add_argument("--output-dir", type=Path, required=True)
    scaling.add_argument("--output-name", default="metrics_vs_sample_size.png")
    scaling.add_argument("--impls", default="", help="Comma- or space-separated implementations to plot.")
    scaling.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    scaling.add_argument("--title", default="Benchmark Scaling")
    scaling.set_defaults(func=command_scaling)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
