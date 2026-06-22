#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
DEFAULT_OUTPUT_DIR = BENCH_DIR / "baseline_vs_online_join_scaling"

METRICS = [
    ("wall_seconds", "Mean Wall Time", "seconds", "mean"),
    ("engine_seconds", "Mean Engine Time", "seconds", "mean"),
    ("llm_prompts", "Mean LLM Prompts", "count", "mean"),
    ("structured_candidates", "Mean Structured Candidates", "count", "mean"),
    ("semantic_rows", "Mean Semantic Rows", "count", "mean"),
    ("join_rows", "Mean Join Rows", "count", "mean"),
    ("result_rows", "Mean Result Rows", "count", "mean"),
]

COLORS = {
    "baseline": "#2563eb",
    "online_join": "#dc2626",
}


def load_metrics(sizes: list[int], run_prefix: str) -> pd.DataFrame:
    frames = []
    missing = []
    for size in sizes:
        path = BENCH_DIR / f"{run_prefix}_{size}" / "metrics.csv"
        if not path.exists():
            matches = sorted(
                BENCH_DIR.glob(f"{run_prefix}_{size}_*/metrics.csv"),
                key=lambda candidate: candidate.parent.stat().st_mtime,
                reverse=True,
            )
            if matches:
                path = matches[0]
            else:
                missing.append(str(path.relative_to(ROOT)))
                continue
        frame = pd.read_csv(path)
        frame.insert(0, "sample_size", size)
        frame.insert(1, "run_name", path.parent.name)
        frames.append(frame)

    if missing:
        raise FileNotFoundError("Missing metrics files:\n" + "\n".join(missing))
    if not frames:
        raise ValueError("No metrics files found")

    data = pd.concat(frames, ignore_index=True)
    bad = data[data["exit_code"] != 0]
    if not bad.empty:
        cols = ["sample_size", "project", "query_id", "exit_code", "log_path"]
        raise ValueError("Non-zero benchmark exits:\n" + bad[cols].to_string(index=False))
    return data


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "wall_seconds",
        "engine_seconds",
        "llm_prompts",
        "structured_candidates",
        "semantic_rows",
        "join_rows",
        "result_rows",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    grouped = data.groupby(["sample_size", "project"], as_index=False)
    summary = grouped.agg(
        queries=("query_id", "nunique"),
        total_wall_seconds=("wall_seconds", "sum"),
        mean_wall_seconds=("wall_seconds", "mean"),
        median_wall_seconds=("wall_seconds", "median"),
        total_engine_seconds=("engine_seconds", "sum"),
        mean_engine_seconds=("engine_seconds", "mean"),
        total_llm_prompts=("llm_prompts", "sum"),
        mean_llm_prompts=("llm_prompts", "mean"),
        mean_structured_candidates=("structured_candidates", "mean"),
        mean_semantic_rows=("semantic_rows", "mean"),
        mean_join_rows=("join_rows", "mean"),
        mean_result_rows=("result_rows", "mean"),
    )
    return summary.sort_values(["sample_size", "project"])


def plot(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), sharex=True)
    axes_flat = axes.flatten()

    for ax, (source_column, title, ylabel, _) in zip(axes_flat, METRICS):
        if source_column == "wall_seconds":
            column = "mean_wall_seconds"
        elif source_column == "engine_seconds":
            column = "mean_engine_seconds"
        elif source_column == "llm_prompts":
            column = "mean_llm_prompts"
        else:
            column = f"mean_{source_column}"

        for project in ["baseline", "online_join"]:
            subset = summary[summary["project"] == project].sort_values("sample_size")
            if subset.empty:
                continue
            ax.plot(
                subset["sample_size"],
                subset[column],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=COLORS.get(project),
                label=project,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e5e7eb", linewidth=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)

    axes_flat[-1].axis("off")
    for ax in axes_flat[:-1]:
        ax.set_xlabel("sample_size")
        ax.set_xscale("log")
        ax.set_xticks(sorted(summary["sample_size"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    fig.suptitle("Baseline vs Online Join Scaling", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.975), h_pad=3.0, w_pad=2.5)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate baseline vs online_join scaling benchmark metrics.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 100, 200])
    parser.add_argument("--run-prefix", default="baseline_vs_online_join")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    data = load_metrics(args.sizes, args.run_prefix)
    summary = summarize(data.copy())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = args.output_dir / "detailed_metrics.csv"
    summary_path = args.output_dir / "summary.csv"
    plot_path = args.output_dir / "metrics_vs_sample_size.png"

    data.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot(summary, plot_path)

    print(f"Detailed metrics saved to: {detailed_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
