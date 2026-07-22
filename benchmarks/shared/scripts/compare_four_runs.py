#!/usr/bin/env python3
"""Create publication-ready plots from matched SUQL and Trummer runs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = [
    "suql_baseline",
    "suql_v1_two_level_cascade",
    "trummer_baseline_adaptive_block_join",
    "trummer_v1_structured_two_level_cascade",
]
SHORT = {
    "suql_baseline": "SUQL baseline",
    "suql_v1_two_level_cascade": "SUQL V1",
    "trummer_baseline_adaptive_block_join": "Trummer baseline",
    "trummer_v1_structured_two_level_cascade": "Trummer V1",
}
QUALITY_COLORS = {"precision": "#4C78A8", "recall": "#F58518", "f1": "#6F5BD3"}
TIME_COLORS = {
    "structured_orchestration_seconds": "#B9C1CC",
    "cheap_seconds": "#54A76B",
    "expensive_seconds": "#D86727",
}


def load_pair(suql_dir: Path, trummer_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    suql_aggregate = pd.read_csv(suql_dir / "aggregate.csv")
    suql_questions = pd.read_csv(suql_dir / "comparison.csv")
    suql_aggregate = suql_aggregate[suql_aggregate["implementation"].str.startswith("suql")]
    suql_questions = suql_questions[suql_questions["implementation"].str.startswith("suql")]
    trummer_aggregate = pd.read_csv(trummer_dir / "aggregate.csv")
    trummer_questions = pd.read_csv(trummer_dir / "comparison.csv")
    trummer_aggregate = trummer_aggregate[trummer_aggregate["implementation"].str.startswith("trummer")]
    trummer_questions = trummer_questions[trummer_questions["implementation"].str.startswith("trummer")]
    aggregate = pd.concat([suql_aggregate, trummer_aggregate], ignore_index=True)
    questions = pd.concat([suql_questions, trummer_questions], ignore_index=True)
    aggregate["implementation"] = pd.Categorical(aggregate["implementation"], ORDER, ordered=True)
    questions["implementation"] = pd.Categorical(questions["implementation"], ORDER, ordered=True)
    return aggregate.sort_values("implementation"), questions.sort_values(["question_index", "implementation"])


def plot_aggregate_quality(frame: pd.DataFrame, path: Path) -> None:
    labels = [SHORT[str(item)] for item in frame["implementation"]]
    x = np.arange(len(frame)); width = 0.24
    fig, ax = plt.subplots(figsize=(11, 6.4))
    for offset, metric in zip((-width, 0, width), ("precision", "recall", "f1")):
        bars = ax.bar(x + offset, frame[metric], width, label=metric.title(), color=QUALITY_COLORS[metric])
        for bar, value in zip(bars, frame[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, value + .018, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_title("Four solutions: mean quality over 5 questions (1 repetition)")
    ax.set_ylabel("Score (higher is better)"); ax.set_ylim(0, 1.08)
    ax.set_xticks(x, labels); ax.legend(ncols=3, loc="upper center")
    ax.grid(axis="y", alpha=.22); fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_question_f1(frame: pd.DataFrame, path: Path) -> None:
    pivot = frame.pivot(index="question", columns="implementation", values="f1").reindex(columns=ORDER)
    x = np.arange(len(pivot)); width = .19
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = ["#7AA6C2", "#315A7D", "#E8A36A", "#A64B20"]
    for index, (implementation, color) in enumerate(zip(ORDER, colors)):
        values = pivot[implementation].to_numpy(float)
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=SHORT[implementation], color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+.014, f"{value:.2f}", ha="center", fontsize=8, rotation=90)
    ax.set_title("F1 by benchmark question")
    ax.set_ylabel("F1 (higher is better)"); ax.set_ylim(0, 1.08)
    ax.set_xticks(x, [str(item).upper() for item in pivot.index])
    ax.legend(ncols=2); ax.grid(axis="y", alpha=.22)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def stage_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    measured_semantic = result["cheap_seconds"] + result["expensive_seconds"]
    result["structured_orchestration_seconds"] = (result["wall_seconds"] - measured_semantic).clip(lower=0)
    return result


def plot_time(frame: pd.DataFrame, path: Path) -> None:
    frame = stage_frame(frame)
    labels = [SHORT[str(item)] for item in frame["implementation"]]
    stages = ["structured_orchestration_seconds", "cheap_seconds", "expensive_seconds"]
    stage_labels = [
        "Structured + orchestration residual",
        "Cheap semantic prompts",
        "Expensive semantic prompts",
    ]
    y = np.arange(len(frame))
    fig, (absolute, percent) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.25, 1]})
    left_abs = np.zeros(len(frame)); left_pct = np.zeros(len(frame))
    totals = frame["wall_seconds"].to_numpy(float)
    for stage, label in zip(stages, stage_labels):
        values = frame[stage].to_numpy(float)
        percentages = np.divide(values * 100, totals, out=np.zeros_like(values), where=totals > 0)
        absolute.barh(y, values, left=left_abs, label=label, color=TIME_COLORS[stage])
        percent.barh(y, percentages, left=left_pct, label=label, color=TIME_COLORS[stage])
        for index, (value, pct) in enumerate(zip(values, percentages)):
            if value >= 20 and pct >= 3:
                absolute.text(left_abs[index] + value/2, index, f"{value:.1f}s\n{pct:.1f}%", ha="center", va="center", fontsize=8)
            if pct >= 3:
                percent.text(left_pct[index] + pct/2, index, f"{pct:.1f}%", ha="center", va="center", fontsize=9)
        left_abs += values; left_pct += percentages
    for index, total in enumerate(totals):
        absolute.text(total + max(totals)*.012, index, f"{total:.1f}s", va="center", fontsize=9)
    absolute.set_title("Mean wall time and measured stages")
    absolute.set_xlabel("Seconds per question (lower is better)")
    absolute.set_yticks(y, labels); absolute.invert_yaxis(); absolute.grid(axis="x", alpha=.2)
    percent.set_title("Stage share of recorded wall time")
    percent.set_xlabel("Percent of wall time"); percent.set_xlim(0, 100)
    percent.set_yticks(y, labels); percent.invert_yaxis(); percent.grid(axis="x", alpha=.2)
    handles, legend_labels = absolute.get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncols=3, loc="upper center", bbox_to_anchor=(.5, .985))
    fig.suptitle("Four solutions: timing composition", y=1.04, fontsize=15)
    fig.text(
        .5, .015,
        "Timing caveat: structured execution was not independently instrumented. The gray slice is only wall time "
        "remaining after measured model calls; Trummer V1 structured pruning ran outside its recorded timer.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, .05, 1, .93)); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_calls(frame: pd.DataFrame, path: Path) -> None:
    labels = [SHORT[str(item)] for item in frame["implementation"]]
    y = np.arange(len(frame))
    cheap = frame["cheap_calls"].to_numpy(float)
    expensive = frame["expensive_calls"].to_numpy(float)
    totals = frame["llm_calls"].to_numpy(float)
    cheap_pct = np.divide(cheap * 100, totals, out=np.zeros_like(cheap), where=totals > 0)
    expensive_pct = np.divide(expensive * 100, totals, out=np.zeros_like(expensive), where=totals > 0)
    fig, (absolute, percent) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.25, 1]})
    absolute.barh(y, cheap, label="Cheap semantic prompts", color=TIME_COLORS["cheap_seconds"])
    absolute.barh(y, expensive, left=cheap, label="Expensive semantic prompts", color=TIME_COLORS["expensive_seconds"])
    percent.barh(y, cheap_pct, label="Cheap semantic prompts", color=TIME_COLORS["cheap_seconds"])
    percent.barh(y, expensive_pct, left=cheap_pct, label="Expensive semantic prompts", color=TIME_COLORS["expensive_seconds"])
    for index, total in enumerate(totals):
        absolute.text(total + max(totals)*.012, index, f"{total:.1f}", va="center", fontsize=9)
        if cheap_pct[index] >= 5:
            percent.text(cheap_pct[index]/2, index, f"{cheap[index]:.1f}\n{cheap_pct[index]:.1f}%", ha="center", va="center", fontsize=9)
        if expensive_pct[index] >= 5:
            percent.text(cheap_pct[index]+expensive_pct[index]/2, index, f"{expensive[index]:.1f}\n{expensive_pct[index]:.1f}%", ha="center", va="center", fontsize=9)
    absolute.set_title("Mean model calls per question")
    absolute.set_xlabel("LLM calls (lower is better)")
    absolute.set_yticks(y, labels); absolute.invert_yaxis(); absolute.grid(axis="x", alpha=.2)
    percent.set_title("Cheap/expensive share of calls")
    percent.set_xlabel("Percent of calls"); percent.set_xlim(0, 100)
    percent.set_yticks(y, labels); percent.invert_yaxis(); percent.grid(axis="x", alpha=.2)
    handles, legend_labels = absolute.get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncols=2, loc="upper center", bbox_to_anchor=(.5, .98))
    fig.suptitle("Four solutions: LLM call composition", y=1.035, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_structured_semantic_time(frame: pd.DataFrame, path: Path) -> None:
    """Show measured semantic time and the unavailable structured stage honestly."""
    labels = [SHORT[str(item)] for item in frame["implementation"]]
    y = np.arange(len(frame))
    semantic = (frame["cheap_seconds"] + frame["expensive_seconds"]).to_numpy(float)
    fig, (seconds_ax, status_ax) = plt.subplots(
        1, 2, figsize=(15, 6.8), gridspec_kw={"width_ratios": [1.45, .8]},
    )
    bars = seconds_ax.barh(y, semantic, color="#D86727", label="Semantic filter (measured model time)")
    for bar, value in zip(bars, semantic):
        seconds_ax.text(value + max(semantic)*.012, bar.get_y()+bar.get_height()/2, f"{value:.1f}s", va="center", fontsize=9)
    seconds_ax.set_title("Measured semantic-filter time")
    seconds_ax.set_xlabel("Mean seconds per question")
    seconds_ax.set_yticks(y, labels); seconds_ax.invert_yaxis(); seconds_ax.grid(axis="x", alpha=.2)
    status_ax.barh(y, np.ones(len(frame)), color="#E2E5E9", edgecolor="#77808A", hatch="///")
    for index in range(len(frame)):
        status_ax.text(.5, index, "N/A\nnot instrumented", ha="center", va="center", fontsize=9)
    status_ax.set_title("Structured-filter time")
    status_ax.set_xlim(0, 1); status_ax.set_xticks([])
    status_ax.set_yticks(y, labels); status_ax.invert_yaxis()
    fig.suptitle("Structured versus semantic filtering stages", fontsize=15)
    fig.text(
        .5, .015,
        "Structured time cannot be reconstructed from these runs: SUQL includes it inside query execution, "
        "and Trummer V1 performs structured pruning outside its recorded timer. N/A is not zero.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, .05, 1, .94)); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_recall_time_calls(frame: pd.DataFrame, path: Path) -> None:
    implementations = [str(item) for item in frame["implementation"]]
    times = frame["wall_seconds"].to_numpy(float)
    recalls = frame["recall"].to_numpy(float)
    calls = frame["llm_calls"].to_numpy(float)
    pareto = []
    for index in range(len(frame)):
        dominated = any(
            other != index
            and times[other] <= times[index]
            and recalls[other] >= recalls[index]
            and (times[other] < times[index] or recalls[other] > recalls[index])
            for other in range(len(frame))
        )
        if not dominated:
            pareto.append(index)
    sizes = 180 + 7 * calls
    colors = ["#7AA6C2", "#315A7D", "#E8A36A", "#A64B20"]
    fig, ax = plt.subplots(figsize=(11, 7))
    for index, (implementation, color) in enumerate(zip(implementations, colors)):
        ax.scatter(
            times[index], recalls[index], s=sizes[index], color=color, alpha=.82,
            edgecolor="#111111" if index in pareto else "white",
            linewidth=3 if index in pareto else 1.5,
        )
        ax.annotate(
            f"{SHORT[implementation]}\n{recalls[index]:.3f} recall · {times[index]:.1f}s · {calls[index]:.1f} calls",
            (times[index], recalls[index]), xytext=(9, 9), textcoords="offset points", fontsize=9,
        )
    frontier = sorted(pareto, key=lambda item: times[item])
    if len(frontier) > 1:
        ax.plot(times[frontier], recalls[frontier], color="#333333", linestyle="--", alpha=.55, label="Pareto frontier")
    legend_calls = [10, 50, 100]
    bubble_handles = [ax.scatter([], [], s=180+7*value, color="#999999", alpha=.45, edgecolor="white") for value in legend_calls]
    frontier_handle = plt.Line2D([], [], color="#333333", linestyle="--")
    ax.legend([frontier_handle, *bubble_handles], ["Pareto frontier", *[f"{value} calls" for value in legend_calls]], loc="lower right")
    ax.set_title("Recall–time trade-off (bubble size = mean LLM calls)")
    ax.set_xlabel("Mean wall time per question in seconds (lower is better)")
    ax.set_ylabel("Mean recall (higher is better)")
    ax.set_xlim(left=0); ax.set_ylim(0, 1); ax.grid(alpha=.22)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suql-dir", type=Path, required=True)
    parser.add_argument("--trummer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate, questions = load_pair(args.suql_dir, args.trummer_dir)
    aggregate = stage_frame(aggregate)
    aggregate.to_csv(args.output_dir / "four_solutions_aggregate.csv", index=False)
    questions.to_csv(args.output_dir / "four_solutions_per_question.csv", index=False)
    plot_aggregate_quality(aggregate, args.output_dir / "01_quality_metrics.png")
    plot_question_f1(questions, args.output_dir / "02_quality_f1_by_question.png")
    plot_time(aggregate, args.output_dir / "03_time_by_stage.png")
    plot_calls(aggregate, args.output_dir / "04_calls_by_stage.png")
    plot_recall_time_calls(aggregate, args.output_dir / "05_recall_time_calls_tradeoff.png")
    plot_structured_semantic_time(aggregate, args.output_dir / "06_structured_vs_semantic_time.png")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
