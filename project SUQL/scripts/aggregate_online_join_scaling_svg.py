#!/usr/bin/env python3
"""Aggregate baseline vs online_join scaling metrics and write an SVG plot.

This is a no-matplotlib fallback for low-memory cluster login nodes. It uses
only the Python standard library and writes:
  - detailed_metrics.csv
  - summary.csv
  - metrics_vs_sample_size.svg
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
DEFAULT_OUTPUT_DIR = BENCH_DIR / "baseline_vs_online_join_scaling"

METRICS = [
    ("mean_wall_seconds", "Mean Wall Time", "seconds"),
    ("mean_engine_seconds", "Mean Engine Time", "seconds"),
    ("mean_llm_prompts", "Mean LLM Prompts", "count"),
    ("mean_structured_candidates", "Mean Structured Candidates", "count"),
    ("mean_semantic_rows", "Mean Semantic Rows", "count"),
    ("mean_join_rows", "Mean Join Rows", "count"),
    ("mean_result_rows", "Mean Result Rows", "count"),
]

COLORS = {
    "baseline": "#2563eb",
    "online_join": "#dc2626",
}


def metrics_path_for(size: int, run_prefix: str) -> Path:
    exact = BENCH_DIR / f"{run_prefix}_{size}" / "metrics.csv"
    if exact.exists():
        return exact

    matches = sorted(
        BENCH_DIR.glob(f"{run_prefix}_{size}_*/metrics.csv"),
        key=lambda candidate: candidate.parent.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No metrics.csv found for size {size} with prefix {run_prefix}")


def as_float(value: str) -> float:
    try:
        if value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_metrics(sizes: list[int], run_prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for size in sizes:
        path = metrics_path_for(size, run_prefix)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["sample_size"] = str(size)
                row["run_name"] = path.parent.name
                rows.append(row)

    failures = [
        row
        for row in rows
        if int(as_float(row.get("exit_code", "-1"))) != 0
    ]
    if failures:
        brief = "\n".join(
            f"size={row['sample_size']} project={row.get('project')} "
            f"query={row.get('query_id')} exit={row.get('exit_code')} log={row.get('log_path')}"
            for row in failures[:20]
        )
        raise RuntimeError(f"Non-zero benchmark exits:\n{brief}")

    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    numeric = [
        "wall_seconds",
        "engine_seconds",
        "llm_prompts",
        "structured_candidates",
        "semantic_rows",
        "join_rows",
        "result_rows",
    ]
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["sample_size"], row["project"])].append(row)

    summary: list[dict[str, str]] = []
    for (sample_size, project), group_rows in sorted(
        groups.items(), key=lambda item: (int(item[0][0]), item[0][1])
    ):
        out: dict[str, str] = {
            "sample_size": sample_size,
            "project": project,
            "queries": str(len({row.get("query_id", "") for row in group_rows})),
        }
        for column in numeric:
            values = [as_float(row.get(column, "")) for row in group_rows]
            total = sum(values)
            mean = total / len(values) if values else 0.0
            out[f"total_{column}"] = f"{total:.6g}"
            out[f"mean_{column}"] = f"{mean:.6g}"
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def scale_x(size: int, sizes: list[int], left: float, width: float) -> float:
    if len(set(sizes)) == 1:
        return left + width / 2
    logs = [math.log10(size) for size in sizes]
    lo, hi = min(logs), max(logs)
    return left + ((math.log10(size) - lo) / (hi - lo)) * width


def scale_y(value: float, max_value: float, top: float, height: float) -> float:
    if max_value <= 0:
        return top + height
    return top + height - (value / max_value) * height


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="#111827">{html.escape(text)}</text>'
    )


def plot_svg(summary: list[dict[str, str]], sizes: list[int], output: Path) -> None:
    panel_w = 520
    panel_h = 280
    gap_x = 50
    gap_y = 70
    margin_left = 70
    margin_top = 70
    plot_left_pad = 58
    plot_right_pad = 18
    plot_top_pad = 36
    plot_bottom_pad = 54
    cols = 2
    rows = 4
    svg_w = margin_left * 2 + cols * panel_w + (cols - 1) * gap_x
    svg_h = margin_top + rows * panel_h + (rows - 1) * gap_y + 50

    by_project_metric: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in summary:
        sample_size = int(row["sample_size"])
        project = row["project"]
        for metric, _, _ in METRICS:
            by_project_metric[(project, metric)].append((sample_size, as_float(row.get(metric, "0"))))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(svg_w / 2, 32, "Baseline vs Online Join Scaling", 20, "middle", "700"),
        svg_text(svg_w / 2, 54, "Mean metric value over benchmark queries", 12, "middle"),
    ]

    for index, (metric, title, ylabel) in enumerate(METRICS):
        row_i = index // cols
        col_i = index % cols
        x0 = margin_left + col_i * (panel_w + gap_x)
        y0 = margin_top + row_i * (panel_h + gap_y)
        plot_x = x0 + plot_left_pad
        plot_y = y0 + plot_top_pad
        plot_w = panel_w - plot_left_pad - plot_right_pad
        plot_h = panel_h - plot_top_pad - plot_bottom_pad

        values = [
            value
            for project in ("baseline", "online_join")
            for _, value in by_project_metric.get((project, metric), [])
        ]
        max_value = max(values) if values else 1.0
        if max_value <= 0:
            max_value = 1.0
        max_value *= 1.08

        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#ffffff"/>')
        parts.append(svg_text(x0 + panel_w / 2, y0 + 18, title, 14, "middle", "700"))
        parts.append(svg_text(x0 + 4, y0 + panel_h / 2, ylabel, 11, "start"))

        for frac in (0, 0.25, 0.5, 0.75, 1):
            y = plot_y + plot_h - frac * plot_h
            label = max_value * frac
            parts.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            parts.append(svg_text(plot_x - 8, y + 4, f"{label:.0f}", 10, "end"))

        parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#374151"/>')
        parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#374151"/>')

        for size in sizes:
            x = scale_x(size, sizes, plot_x, plot_w)
            parts.append(f'<line x1="{x:.1f}" y1="{plot_y + plot_h}" x2="{x:.1f}" y2="{plot_y + plot_h + 5}" stroke="#374151"/>')
            parts.append(svg_text(x, plot_y + plot_h + 22, str(size), 10, "middle"))

        for project in ("baseline", "online_join"):
            points = sorted(by_project_metric.get((project, metric), []))
            if not points:
                continue
            color = COLORS[project]
            coords = [
                (
                    scale_x(sample_size, sizes, plot_x, plot_w),
                    scale_y(value, max_value, plot_y, plot_h),
                )
                for sample_size, value in points
            ]
            path_data = " ".join(
                ("M" if idx == 0 else "L") + f"{x:.1f},{y:.1f}"
                for idx, (x, y) in enumerate(coords)
            )
            parts.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for x, y in coords:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')

        legend_x = x0 + panel_w - 155
        legend_y = y0 + 38
        for offset, project in enumerate(("baseline", "online_join")):
            y = legend_y + offset * 18
            color = COLORS[project]
            parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
            parts.append(f'<circle cx="{legend_x + 11}" cy="{y}" r="3.5" fill="{color}"/>')
            parts.append(svg_text(legend_x + 30, y + 4, project, 11))

    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate baseline vs online_join scaling metrics without matplotlib.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[200, 500, 1000])
    parser.add_argument("--run-prefix", default="aker_data_sample")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = load_metrics(args.sizes, args.run_prefix)
    summary = summarize(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = args.output_dir / "detailed_metrics.csv"
    summary_path = args.output_dir / "summary.csv"
    plot_path = args.output_dir / "metrics_vs_sample_size.svg"

    write_csv(detailed_path, rows)
    write_csv(summary_path, summary)
    plot_svg(summary, args.sizes, plot_path)

    print(f"Detailed metrics saved to: {detailed_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"SVG plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
