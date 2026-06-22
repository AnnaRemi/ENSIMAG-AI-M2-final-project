#!/usr/bin/env python3
"""Analyze Stage 2 experiment_#1 sample-size benchmark runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_DIR = ROOT / "Stage_2" / "benchmarks"
DEFAULT_OUTPUT_DIR = DEFAULT_BENCHMARK_DIR / "experiment_#1_analysis"

PROJECT_COLORS = {
    "baseline": "#238bb8",
    "stage2": "#f08a24",
}

NUMERIC_COLUMNS = [
    "wall_seconds",
    "engine_seconds",
    "llm_full_calls",
    "llm_prompts_issued",
    "structured_candidates",
    "semantic_rows",
    "join_rows",
    "result_rows",
    "cheap_score_calls",
    "cheap_score_failures",
    "expensive_full_calls",
    "cheap_early_accept",
    "cheap_early_reject",
    "cheap_skipped",
    "cheap_disabled",
]


def sample_size_from_path(path: Path) -> int:
    match = re.search(r"experiment_#1_(\d+)", str(path))
    if not match:
        raise ValueError(f"Cannot extract sample size from {path}")
    return int(match.group(1))


def load_runs(benchmark_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metrics_path in sorted(
        benchmark_dir.glob("experiment_#1_*/metrics.csv"),
        key=sample_size_from_path,
    ):
        frame = pd.read_csv(metrics_path)
        frame["sample_size"] = sample_size_from_path(metrics_path)
        frame["experiment"] = metrics_path.parent.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No experiment_#1_* metrics.csv files found under {benchmark_dir}")

    data = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["exit_code"] = pd.to_numeric(data["exit_code"], errors="coerce").fillna(-1).astype(int)
    return data


def aggregate_successful(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["sample_size", "project"], as_index=False)
    ok_counts = grouped.agg(
        ok_queries=("exit_code", lambda values: int((values == 0).sum())),
        failed_queries=("exit_code", lambda values: int((values != 0).sum())),
    )

    successful = data[data["exit_code"] == 0].copy()
    sums = (
        successful.groupby(["sample_size", "project"], as_index=False)[NUMERIC_COLUMNS]
        .sum(numeric_only=True)
        .merge(ok_counts, on=["sample_size", "project"], how="right")
        .fillna(0)
        .sort_values(["sample_size", "project"])
    )

    sums["wall_seconds_per_structured_candidate"] = (
        sums["wall_seconds"] / sums["structured_candidates"].replace(0, pd.NA)
    )
    sums["engine_seconds_per_structured_candidate"] = (
        sums["engine_seconds"] / sums["structured_candidates"].replace(0, pd.NA)
    )
    return sums


def paired_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["project"] == "baseline"].set_index("sample_size")
    stage2 = summary[summary["project"] == "stage2"].set_index("sample_size")
    common = baseline.index.intersection(stage2.index)

    rows = []
    for sample_size in common:
        base = baseline.loc[sample_size]
        stage = stage2.loc[sample_size]
        rows.append(
            {
                "sample_size": sample_size,
                "baseline_wall_seconds": base["wall_seconds"],
                "stage2_wall_seconds": stage["wall_seconds"],
                "wall_seconds_delta_stage2_minus_baseline": stage["wall_seconds"] - base["wall_seconds"],
                "wall_speedup_baseline_over_stage2": base["wall_seconds"] / stage["wall_seconds"],
                "baseline_engine_seconds": base["engine_seconds"],
                "stage2_engine_seconds": stage["engine_seconds"],
                "engine_seconds_delta_stage2_minus_baseline": stage["engine_seconds"] - base["engine_seconds"],
                "engine_speedup_baseline_over_stage2": base["engine_seconds"] / stage["engine_seconds"],
                "baseline_prompts": base["llm_prompts_issued"],
                "stage2_prompts": stage["llm_prompts_issued"],
                "prompt_delta_stage2_minus_baseline": stage["llm_prompts_issued"] - base["llm_prompts_issued"],
                "baseline_expensive_full_calls": base["expensive_full_calls"],
                "stage2_expensive_full_calls": stage["expensive_full_calls"],
                "expensive_full_calls_saved": base["expensive_full_calls"] - stage["expensive_full_calls"],
                "stage2_cheap_score_calls": stage["cheap_score_calls"],
                "stage2_cheap_rejects": stage["cheap_early_reject"],
                "structured_candidates": base["structured_candidates"],
                "baseline_result_rows": base["result_rows"],
                "stage2_result_rows": stage["result_rows"],
            }
        )

    return pd.DataFrame(rows).sort_values("sample_size")


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#e5e9f0", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_latency(summary: pd.DataFrame, comparison: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    for metric, title, ax in [
        ("wall_seconds", "Total Wall Time", axes[0]),
        ("engine_seconds", "Total Engine Time", axes[1]),
    ]:
        for project in ["baseline", "stage2"]:
            project_data = summary[summary["project"] == project]
            ax.plot(
                project_data["sample_size"],
                project_data[metric],
                marker="o",
                linewidth=2.4,
                color=PROJECT_COLORS[project],
                label=project,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("sample size")
        ax.set_ylabel("seconds")
        ax.legend()
        style_axes(ax)

    axes[2].axhline(1.0, color="#30343b", linewidth=1.2, linestyle="--")
    axes[2].plot(
        comparison["sample_size"],
        comparison["wall_speedup_baseline_over_stage2"],
        marker="o",
        linewidth=2.4,
        color="#4f46e5",
    )
    axes[2].set_title("Wall-Time Speedup", fontweight="bold")
    axes[2].set_xlabel("sample size")
    axes[2].set_ylabel("baseline / stage2")
    style_axes(axes[2])

    fig.suptitle("experiment_#1: Latency Scaling", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_llm_work(summary: pd.DataFrame, comparison: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    for project in ["baseline", "stage2"]:
        project_data = summary[summary["project"] == project]
        axes[0].plot(
            project_data["sample_size"],
            project_data["llm_prompts_issued"],
            marker="o",
            linewidth=2.4,
            color=PROJECT_COLORS[project],
            label=project,
        )
        axes[1].plot(
            project_data["sample_size"],
            project_data["expensive_full_calls"],
            marker="o",
            linewidth=2.4,
            color=PROJECT_COLORS[project],
            label=project,
        )

    axes[2].bar(
        comparison["sample_size"],
        comparison["expensive_full_calls_saved"],
        width=70,
        color="#16a34a",
        label="expensive calls saved",
    )
    axes[2].plot(
        comparison["sample_size"],
        comparison["stage2_cheap_score_calls"],
        marker="o",
        linewidth=2.4,
        color="#7c3aed",
        label="stage2 cheap score calls",
    )

    titles = ["Total LLM Prompts", "Expensive Full Calls", "Saved Expensive Calls vs Cheap Calls"]
    ylabels = ["count", "count", "count"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("sample size")
        ax.set_ylabel(ylabel)
        ax.legend()
        style_axes(ax)

    fig.suptitle("experiment_#1: LLM Workload", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_scaling(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    baseline = summary[summary["project"] == "baseline"]
    axes[0].plot(
        baseline["sample_size"],
        baseline["structured_candidates"],
        marker="o",
        linewidth=2.4,
        color="#0f766e",
        label="structured candidates",
    )
    axes[0].plot(
        baseline["sample_size"],
        baseline["semantic_rows"],
        marker="o",
        linewidth=2.4,
        color="#dc2626",
        label="semantic rows",
    )

    for project in ["baseline", "stage2"]:
        project_data = summary[summary["project"] == project]
        axes[1].plot(
            project_data["sample_size"],
            project_data["wall_seconds_per_structured_candidate"],
            marker="o",
            linewidth=2.4,
            color=PROJECT_COLORS[project],
            label=project,
        )
        axes[2].plot(
            project_data["sample_size"],
            project_data["result_rows"],
            marker="o",
            linewidth=2.4,
            color=PROJECT_COLORS[project],
            label=project,
        )

    titles = ["Candidate Growth", "Wall Time per Structured Candidate", "Returned Result Rows"]
    ylabels = ["count", "seconds / candidate", "count"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("sample size")
        ax.set_ylabel(ylabel)
        ax.legend()
        style_axes(ax)

    fig.suptitle("experiment_#1: Retrieval Scale Effects", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment_#1 sample-size benchmark runs.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_runs(args.benchmark_dir)
    summary = aggregate_successful(data)
    comparison = paired_comparison(summary)

    summary.to_csv(args.output_dir / "summary_by_sample_and_project.csv", index=False)
    comparison.to_csv(args.output_dir / "paired_comparison.csv", index=False)

    plot_latency(summary, comparison, args.output_dir / "latency_scaling.png")
    plot_llm_work(summary, comparison, args.output_dir / "llm_workload.png")
    plot_candidate_scaling(summary, args.output_dir / "retrieval_scale_effects.png")

    print(f"Wrote {args.output_dir / 'summary_by_sample_and_project.csv'}")
    print(f"Wrote {args.output_dir / 'paired_comparison.csv'}")
    print(f"Wrote {args.output_dir / 'latency_scaling.png'}")
    print(f"Wrote {args.output_dir / 'llm_workload.png'}")
    print(f"Wrote {args.output_dir / 'retrieval_scale_effects.png'}")


if __name__ == "__main__":
    main()
