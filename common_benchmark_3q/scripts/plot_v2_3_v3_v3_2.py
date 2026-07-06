#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["V2_3", "V3", "V3_2"]
METHOD_CONTEXT = {
    "V2_3": "V2_3: batch-wise cascade without structured pruning",
    "V3": "V3: SUQL-style structured pruning + row-wise cascade",
    "V3_2": "V3_2: SUQL-style structured pruning + batch-wise cascade",
}
METHOD_DIRS = {
    "V2_3": "heterogen_v2_3",
    "V3": "heterogen_v3",
    "V3_2": "heterogen_v3_2",
}
METRIC_COLORS = {
    "Precision": "#4c78a8",
    "Recall": "#f58518",
    "F1": "#6f5bd3",
}
CHEAP_COLOR = "#4c9f70"
EXPENSIVE_COLOR = "#d95f02"


def read_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def method_threshold_label(metrics: dict) -> str:
    learned = metrics.get("learned_confidence_threshold")
    routing = metrics.get("routing_confidence_threshold")
    manual = metrics.get("manual_confidence_threshold")
    target = metrics.get("cascade_target", 0.9)
    if isinstance(learned, (int, float)):
        return f"learned t={learned:.2f}"
    if isinstance(routing, (int, float)):
        return f"routing t={routing:.2f}"
    if isinstance(manual, (int, float)):
        return f"manual t={manual:.2f}"
    return f"target={float(target):.2f}\n(no learned t)"


def question_title(question_dir: str) -> str:
    parts = question_dir.split("_")
    if len(parts) >= 2 and parts[0] == "question":
        return f"Q{parts[1]}"
    return question_dir


def load_thresholds(output_dir: Path, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        metrics_path = output_dir / row["question"] / METHOD_DIRS[row["method"]] / "run_metrics.json"
        metrics = read_json(metrics_path)
        rows.append(
            {
                "question": row["question"],
                "method": row["method"],
                "threshold_label": method_threshold_label(metrics),
                "learned_confidence_threshold": metrics.get("learned_confidence_threshold"),
                "routing_confidence_threshold": metrics.get("routing_confidence_threshold"),
                "manual_confidence_threshold": metrics.get("manual_confidence_threshold"),
                "cascade_target": metrics.get("cascade_target", 0.9),
                "structured_filters": json.dumps(metrics.get("structured_filters", [])),
            }
        )
    return pd.DataFrame(rows)


def add_question_boxes(ax, questions: list[str], methods: list[str], y: float) -> None:
    group = len(methods)
    for index, question in enumerate(questions):
        left = index * group - 0.5
        right = (index + 1) * group - 0.5
        center = index * group + (group - 1) / 2
        ax.axvline(left, color="#222222", linewidth=1.0)
        ax.text(center, y, question_title(question), ha="center", va="bottom", fontsize=11, fontweight="bold")
        if index == len(questions) - 1:
            ax.axvline(right, color="#222222", linewidth=1.0)


def x_layout(frame: pd.DataFrame, thresholds: pd.DataFrame) -> tuple[list[str], list[str], list[int], list[str]]:
    questions = list(dict.fromkeys(frame["question"].tolist()))
    positions = []
    labels = []
    threshold_lookup = {
        (row.question, row.method): row.threshold_label
        for row in thresholds.itertuples(index=False)
    }
    for q_index, question in enumerate(questions):
        for m_index, method in enumerate(METHOD_ORDER):
            positions.append(q_index * len(METHOD_ORDER) + m_index)
            labels.append(f"{method}\n{threshold_lookup[(question, method)]}")
    return questions, METHOD_ORDER, positions, labels


def question_legend(output_dir: Path, questions: list[str]) -> str:
    benchmark_root = Path(__file__).resolve().parents[1]
    lines = []
    for question in questions:
        benchmark_path = output_dir / question / "benchmark.json"
        if not benchmark_path.exists():
            benchmark_path = benchmark_root / question / "benchmark.json"
        benchmark = read_json(benchmark_path)
        wrapped = textwrap.fill(benchmark["question"], width=62)
        lines.append(f"{question_title(question)}: {wrapped}")
    return "\n\n".join(lines)


def common_axes(ax, output_dir: Path, questions: list[str], methods: list[str], labels: list[str], y: float) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    add_question_boxes(ax, questions, methods, y)
    method_text = "\n".join(METHOD_CONTEXT[m] for m in methods)
    threshold_text = "Threshold labels: learned t = calibrated confidence threshold; target=0.90 means the cascade target hyperparameter was used and no numeric threshold was learned."
    ax.text(
        1.02,
        0.98,
        question_legend(output_dir, questions) + "\n\n" + method_text + "\n\n" + textwrap.fill(threshold_text, width=66),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )


def plot_metrics(output_dir: Path, frame: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    questions, methods, positions, labels = x_layout(frame, thresholds)
    values = []
    for question in questions:
        for method in methods:
            row = frame[(frame["question"] == question) & (frame["method"] == method)].iloc[0]
            values.append(row)

    fig, ax = plt.subplots(figsize=(18, 7))
    width = 0.22
    for offset, (name, column) in zip([-width, 0, width], [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]):
        heights = [row[column] for row in values]
        ax.bar([pos + offset for pos in positions], heights, width=width, label=name, color=METRIC_COLORS[name])

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Quality metrics by question and implementation")
    ax.legend(loc="upper left", ncols=3, frameon=False)
    common_axes(ax, output_dir, questions, methods, labels, 1.04)
    fig.tight_layout(rect=(0, 0, 0.67, 1))
    fig.savefig(output_dir / "metrics_precision_recall_f1.png", dpi=180)
    fig.savefig(output_dir / "metrics_by_question_with_thresholds.png", dpi=180)
    plt.close(fig)


def annotate_stack(ax, x: int, cheap: float, expensive: float, mode: str) -> None:
    total = cheap + expensive
    if total <= 0:
        return
    if mode == "percent":
        cheap_text = f"{cheap / total * 100:.0f}% cheap"
        expensive_text = f"{expensive / total * 100:.0f}% expensive"
    else:
        cheap_text = f"{cheap:.0f} cheap"
        expensive_text = f"{expensive:.0f} expensive"
    if cheap > 0:
        if mode == "count" and cheap < 4:
            ax.text(x - 0.36, cheap / 2, cheap_text, ha="right", va="center", fontsize=7, color="#222222")
        else:
            ax.text(x, cheap / 2, cheap_text, ha="center", va="center", fontsize=7, color="white", rotation=90)
    if expensive > 0:
        if mode == "count" and expensive < 4:
            ax.text(x + 0.36, cheap + expensive / 2, expensive_text, ha="left", va="center", fontsize=7, color="#222222")
        else:
            ax.text(x, cheap + expensive / 2, expensive_text, ha="center", va="center", fontsize=7, color="white", rotation=90)


def plot_stacked(output_dir: Path, frame: pd.DataFrame, thresholds: pd.DataFrame, kind: str) -> None:
    questions, methods, positions, labels = x_layout(frame, thresholds)
    ordered = []
    for question in questions:
        for method in methods:
            ordered.append(frame[(frame["question"] == question) & (frame["method"] == method)].iloc[0])

    if kind == "time":
        cheap_col, expensive_col = "cheap_seconds", "expensive_seconds"
        ylabel = "Mean seconds"
        title = "Mean time split between cheap and expensive calls"
        outfile = "time_bar_plot.png"
        mode = "percent"
    else:
        cheap_col, expensive_col = "cheap_calls", "expensive_calls"
        ylabel = "Mean calls"
        title = "Mean call count split between cheap and expensive calls"
        outfile = "calls_bar_plot.png"
        mode = "count"

    cheap = [row[cheap_col] for row in ordered]
    expensive = [row[expensive_col] for row in ordered]
    totals = [c + e for c, e in zip(cheap, expensive)]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(positions, cheap, width=0.62, color=CHEAP_COLOR, label="Cheap calls")
    ax.bar(positions, expensive, width=0.62, bottom=cheap, color=EXPENSIVE_COLOR, label="Expensive calls")
    for x, c, e in zip(positions, cheap, expensive):
        annotate_stack(ax, x, c, e, mode)

    ax.set_ylim(0, max(totals) * 1.22)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", ncols=2, frameon=False)
    common_axes(ax, output_dir, questions, methods, labels, max(totals) * 1.08)
    fig.tight_layout(rect=(0, 0, 0.67, 1))
    fig.savefig(output_dir / outfile, dpi=180)
    plt.close(fig)


def write_summary(output_dir: Path, frame: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    best_f1 = aggregate.sort_values("macro_f1", ascending=False).iloc[0]
    fastest = aggregate.sort_values("total_wall_seconds", ascending=True).iloc[0]
    fewest_calls = aggregate.sort_values("total_llm_calls", ascending=True).iloc[0]
    lines = [
        "# V2_3 vs V3 vs V3_2 comparison",
        "",
        "Run configuration: qwen3:0.6b cheap model, qwen3:1.7b expensive model, 11 repetitions per question.",
        "",
        f"Best macro F1: {best_f1['method']} ({best_f1['macro_f1']:.3f}).",
        f"Fastest total wall time: {fastest['method']} ({fastest['total_wall_seconds']:.2f}s).",
        f"Fewest total LLM calls: {fewest_calls['method']} ({fewest_calls['total_llm_calls']:.0f} calls).",
        "",
        "Per-question observations:",
    ]
    for question in frame["question"].drop_duplicates():
        subset = frame[frame["question"] == question]
        best = subset.sort_values("f1", ascending=False).iloc[0]
        fastest_q = subset.sort_values("wall_seconds").iloc[0]
        lines.append(
            f"- {question_title(question)}: best F1 is {best['method']} ({best['f1']:.3f}); fastest is {fastest_q['method']} ({fastest_q['wall_seconds']:.2f}s)."
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- V3_2 successfully combines SUQL-style structured pruning with batch-wise cascade.",
            "- V3_2 uses far fewer calls than V3, but it is not always faster when it still sends a large expensive fallback batch.",
            "- On Q2, V3_2 is efficient but loses recall, while V3 keeps better quality after pruning.",
            "- On Q3, V3 learns threshold 2.0 and avoids fallback; V3_2 has fewer calls but keeps an expensive fallback batch.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    frame = pd.read_csv(output_dir / "comparison.csv")
    aggregate = pd.read_csv(output_dir / "aggregate.csv")
    frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
    frame["method"] = pd.Categorical(frame["method"], METHOD_ORDER, ordered=True)
    frame = frame.sort_values(["difficulty", "method"]).reset_index(drop=True)
    thresholds = load_thresholds(output_dir, frame)
    thresholds.to_csv(output_dir / "thresholds_used_by_question.csv", index=False)
    frame.to_csv(output_dir / "comparison_v2_3_v3_v3_2.csv", index=False)
    aggregate[aggregate["method"].isin(METHOD_ORDER)].to_csv(output_dir / "aggregate_v2_3_v3_v3_2.csv", index=False)

    plot_metrics(output_dir, frame, thresholds)
    plot_stacked(output_dir, frame, thresholds, "time")
    plot_stacked(output_dir, frame, thresholds, "calls")
    write_summary(output_dir, frame, aggregate[aggregate["method"].isin(METHOD_ORDER)].copy())


if __name__ == "__main__":
    main()
