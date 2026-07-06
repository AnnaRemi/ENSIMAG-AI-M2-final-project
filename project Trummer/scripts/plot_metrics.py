#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trummer_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [ROOT]
DEFAULT_METRICS = ["elapsed_seconds", "cheap_calls", "expensive_calls", "final_answer_rows"]
SKIP_DIRS = {".git", ".venv", "__pycache__", "data", "logs"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Trummer benchmark metrics from baseline, block-join, and cascade outputs."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Output files or directories to scan. Defaults to the whole Trummer project.",
    )
    parser.add_argument(
        "--impl",
        nargs="+",
        help="Implementation filters. Matches implementation names or path fragments.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric names to plot for bar plots.",
    )
    parser.add_argument(
        "--plot",
        choices=["bar", "scatter"],
        default="bar",
        help="Plot type. Use scatter for a two-metric tradeoff view.",
    )
    parser.add_argument("--x", default="elapsed_seconds", help="Scatter x-axis metric.")
    parser.add_argument("--y", default="final_answer_rows", help="Scatter y-axis metric.")
    parser.add_argument("--size", default=None, help="Optional scatter bubble-size metric.")
    parser.add_argument(
        "--aggregate",
        choices=["mean", "none"],
        default="mean",
        help="Average repeated rows by implementation before plotting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "plots" / "trummer_metrics.png",
        help="Output plot path.",
    )
    parser.add_argument("--csv-output", type=Path, help="Optional normalized metrics CSV path.")
    parser.add_argument("--list-metrics", action="store_true", help="Print available numeric metrics and exit.")
    args = parser.parse_args()

    rows = collect_rows(args.input)
    rows = filter_rows(rows, args.impl)
    if args.aggregate == "mean":
        rows = aggregate_rows(rows)
    if not rows:
        raise SystemExit("No metric rows found for the selected inputs/implementations.")

    if args.list_metrics:
        for metric in available_metrics(rows):
            print(metric)
        return

    if args.csv_output:
        write_table(args.csv_output, rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.plot == "bar":
        plot_bar(rows, args.metrics, args.output)
    else:
        plot_scatter(rows, args.x, args.y, args.size, args.output)
    print(f"Wrote {args.output}")


def collect_rows(inputs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_path in inputs:
        path = input_path.expanduser()
        if path.is_file():
            row = load_metric_file(path)
            if row:
                rows.append(row)
            continue
        if not path.exists():
            raise SystemExit(f"Input path does not exist: {path}")
        for file_path in iter_metric_files(path):
            row = load_metric_file(file_path)
            if row:
                rows.append(row)
    return rows


def iter_metric_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.name == "run_metrics.json":
            yield path
        elif path.name.endswith("_metrics.json"):
            yield path
        elif path.name in {"use_case3_join_stats.csv", "join_stats.csv"}:
            yield path


def load_metric_file(path: Path) -> dict[str, Any] | None:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = flatten_numeric(payload)
        row["implementation"] = payload.get("implementation") or infer_implementation(path)
        row["source"] = str(path)
        row["run"] = path.parent.name
        return row
    if path.suffix == ".csv":
        return load_stats_csv(path)
    return None


def load_stats_csv(path: Path) -> dict[str, Any] | None:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        return None
    numeric_columns: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, value in record.items():
            parsed = parse_float(value)
            if parsed is not None:
                numeric_columns[key].append(parsed)
    row: dict[str, Any] = {
        "implementation": infer_implementation(path),
        "source": str(path),
        "run": path.parent.name,
        "llm_calls": len(records),
        "prompts": len(records),
    }
    for key, values in numeric_columns.items():
        row[key] = sum(values)
        row[f"{key}_mean"] = mean(values)
    if "seconds" in row:
        row.setdefault("elapsed_seconds", row["seconds"])
    if "prompt_tokens" in row or "completion_tokens" in row:
        row["tokens_total"] = float(row.get("prompt_tokens", 0.0)) + float(row.get("completion_tokens", 0.0))
    return row


def flatten_numeric(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    row: dict[str, float] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            row[full_key] = float(value)
        elif isinstance(value, dict):
            row.update(flatten_numeric(value, full_key))
    return row


def infer_implementation(path: Path) -> str:
    parts = path.parts
    for part in parts:
        if part.startswith("heterogen_"):
            return part
        if part == "baseline":
            if path.name.endswith("_metrics.json"):
                return f"baseline_{path.name.removesuffix('_metrics.json')}"
            return "baseline"
    return path.parent.name


def filter_rows(rows: list[dict[str, Any]], filters: list[str] | None) -> list[dict[str, Any]]:
    if not filters:
        return rows
    lowered = [item.lower() for item in filters]
    result = []
    for row in rows:
        haystack = f"{row.get('implementation', '')} {row.get('source', '')}".lower()
        if any(item in haystack for item in lowered):
            result.append(row)
    return result


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["implementation"])].append(row)
    result: list[dict[str, Any]] = []
    for implementation, items in sorted(grouped.items()):
        merged: dict[str, Any] = {"implementation": implementation, "runs": len(items)}
        for metric in available_metrics(items):
            values = [float(item[metric]) for item in items if is_number(item.get(metric))]
            if values:
                merged[metric] = mean(values)
        merged["source"] = "; ".join(str(item.get("source", "")) for item in items)
        result.append(merged)
    return result


def available_metrics(rows: list[dict[str, Any]]) -> list[str]:
    metrics = set()
    for row in rows:
        for key, value in row.items():
            if key not in {"implementation", "source", "run"} and is_number(value):
                metrics.add(key)
    return sorted(metrics)


def plot_bar(rows: list[dict[str, Any]], metrics: list[str], output: Path) -> None:
    missing = [metric for metric in metrics if not any(is_number(row.get(metric)) for row in rows)]
    if missing:
        available = ", ".join(available_metrics(rows))
        raise SystemExit(f"Metric(s) not found: {', '.join(missing)}. Available metrics: {available}")

    labels = [str(row["implementation"]) for row in rows]
    width = 0.8 / max(len(metrics), 1)
    x_positions = list(range(len(labels)))
    fig_width = max(10, len(labels) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    for metric_index, metric in enumerate(metrics):
        offset = (metric_index - (len(metrics) - 1) / 2) * width
        values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
        bars = ax.bar([x + offset for x in x_positions], values, width=width, label=metric)
        for bar, value in zip(bars, values):
            ax.annotate(
                format_value(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if len(labels) > 5 else 0,
            )
    ax.set_title("Trummer Benchmark Metrics")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_scatter(rows: list[dict[str, Any]], x_metric: str, y_metric: str, size_metric: str | None, output: Path) -> None:
    for metric in [x_metric, y_metric, *( [size_metric] if size_metric else [] )]:
        if metric and not any(is_number(row.get(metric)) for row in rows):
            available = ", ".join(available_metrics(rows))
            raise SystemExit(f"Metric not found: {metric}. Available metrics: {available}")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10.colors
    for index, row in enumerate(rows):
        x_value = float(row.get(x_metric, 0.0) or 0.0)
        y_value = float(row.get(y_metric, 0.0) or 0.0)
        bubble = 120.0
        if size_metric:
            bubble = max(60.0, math.sqrt(abs(float(row.get(size_metric, 0.0) or 0.0)) + 1.0) * 70.0)
        ax.scatter(
            [x_value],
            [y_value],
            s=bubble,
            color=colors[index % len(colors)],
            alpha=0.78,
            edgecolor="black",
            linewidth=0.6,
            label=str(row["implementation"]),
        )
        ax.annotate(str(index + 1), (x_value, y_value), ha="center", va="center", fontsize=9)
    ax.set_title("Trummer Metric Tradeoff")
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["implementation"] + [metric for metric in available_metrics(rows) if metric != "implementation"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def format_value(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


if __name__ == "__main__":
    main()
