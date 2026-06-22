#!/usr/bin/env python3
"""Plot baseline vs online_join metrics by benchmark question.

This is a no-matplotlib helper intended for Aker/login-node use. It reads one
baseline-vs-online metrics.csv file and writes a multi-panel SVG where the
x-axis is the benchmark question.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "benchmarks" / "aker_data_sample_200" / "metrics.csv"

METRICS = [
    ("wall_seconds", "Wall Time", "seconds"),
    ("engine_seconds", "Engine Time", "seconds"),
    ("llm_prompts", "LLM Prompts", "count"),
    ("structured_candidates", "Structured Candidates", "count"),
    ("semantic_rows", "Semantic Rows", "count"),
    ("join_rows", "Join Rows", "count"),
    ("result_rows", "Result Rows", "count"),
]

COLORS = {
    "baseline": "#2563eb",
    "online_join": "#dc2626",
}


def as_float(value: object) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No rows found in {path}")

    failures = [row for row in rows if int(as_float(row.get("exit_code", "-1"))) != 0]
    if failures:
        brief = "\n".join(
            f"project={row.get('project')} query={row.get('query_id')} "
            f"exit={row.get('exit_code')} log={row.get('log_path')}"
            for row in failures[:20]
        )
        raise RuntimeError(f"Non-zero benchmark exits:\n{brief}")
    return rows


def question_order(rows: list[dict[str, str]]) -> list[str]:
    ordered = []
    for row in rows:
        query_id = row.get("query_id", "")
        if query_id and query_id not in ordered:
            ordered.append(query_id)
    return ordered


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="#111827">{html.escape(text)}</text>'
    )


def scale_x(index: int, count: int, left: float, width: float) -> float:
    if count <= 1:
        return left + width / 2
    return left + (index / (count - 1)) * width


def scale_y(value: float, max_value: float, top: float, height: float) -> float:
    if max_value <= 0:
        return top + height
    return top + height - (value / max_value) * height


def short_label(query_id: str, index: int) -> str:
    return f"Q{index + 1}"


def write_svg(rows: list[dict[str, str]], output: Path, title: str) -> None:
    query_ids = question_order(rows)
    by_project_query = {
        (row.get("project", ""), row.get("query_id", "")): row
        for row in rows
    }

    panel_w = 560
    panel_h = 300
    gap_x = 48
    gap_y = 76
    margin_left = 70
    margin_top = 92
    plot_left_pad = 62
    plot_right_pad = 22
    plot_top_pad = 38
    plot_bottom_pad = 70
    cols = 2
    panel_rows = math.ceil(len(METRICS) / cols)
    svg_w = margin_left * 2 + cols * panel_w + (cols - 1) * gap_x
    svg_h = margin_top + panel_rows * panel_h + (panel_rows - 1) * gap_y + 70

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(svg_w / 2, 34, title, 20, "middle", "700"),
        svg_text(svg_w / 2, 58, "X-axis: benchmark question; values are per query from metrics.csv", 12, "middle"),
    ]

    question_legend = "  ".join(short_label(query_id, i) + "=" + query_id for i, query_id in enumerate(query_ids))
    parts.append(svg_text(svg_w / 2, 78, question_legend, 10, "middle"))

    for metric_index, (metric, metric_title, ylabel) in enumerate(METRICS):
        row_i = metric_index // cols
        col_i = metric_index % cols
        x0 = margin_left + col_i * (panel_w + gap_x)
        y0 = margin_top + row_i * (panel_h + gap_y)
        plot_x = x0 + plot_left_pad
        plot_y = y0 + plot_top_pad
        plot_w = panel_w - plot_left_pad - plot_right_pad
        plot_h = panel_h - plot_top_pad - plot_bottom_pad

        values = [
            as_float(by_project_query.get((project, query_id), {}).get(metric, ""))
            for project in ("baseline", "online_join")
            for query_id in query_ids
        ]
        max_value = max(values) if values else 1.0
        if max_value <= 0:
            max_value = 1.0
        max_value *= 1.10

        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#ffffff"/>')
        parts.append(svg_text(x0 + panel_w / 2, y0 + 18, metric_title, 14, "middle", "700"))
        parts.append(svg_text(x0 + 4, y0 + panel_h / 2, ylabel, 11))

        for frac in (0, 0.25, 0.5, 0.75, 1):
            y = plot_y + plot_h - frac * plot_h
            label = max_value * frac
            parts.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            parts.append(svg_text(plot_x - 8, y + 4, f"{label:.0f}", 10, "end"))

        parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#374151"/>')
        parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#374151"/>')

        for index, query_id in enumerate(query_ids):
            x = scale_x(index, len(query_ids), plot_x, plot_w)
            parts.append(f'<line x1="{x:.1f}" y1="{plot_y + plot_h}" x2="{x:.1f}" y2="{plot_y + plot_h + 5}" stroke="#374151"/>')
            parts.append(svg_text(x, plot_y + plot_h + 22, short_label(query_id, index), 10, "middle"))

        for project in ("baseline", "online_join"):
            coords = []
            for index, query_id in enumerate(query_ids):
                row = by_project_query.get((project, query_id), {})
                value = as_float(row.get(metric, ""))
                coords.append((scale_x(index, len(query_ids), plot_x, plot_w), scale_y(value, max_value, plot_y, plot_h), value))

            color = COLORS[project]
            path_data = " ".join(
                ("M" if idx == 0 else "L") + f"{x:.1f},{y:.1f}"
                for idx, (x, y, _) in enumerate(coords)
            )
            parts.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for x, y, value in coords:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
                parts.append(svg_text(x, y - 8, f"{value:.0f}", 9, "middle"))

        legend_x = x0 + panel_w - 160
        legend_y = y0 + 40
        for offset, project in enumerate(("baseline", "online_join")):
            y = legend_y + offset * 18
            color = COLORS[project]
            parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
            parts.append(f'<circle cx="{legend_x + 11}" cy="{y}" r="3.5" fill="{color}"/>')
            parts.append(svg_text(legend_x + 30, y + 4, project, 11))

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate question-vs-metric SVG from baseline vs online_join metrics.csv.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Baseline vs Online Join by Question")
    args = parser.parse_args()

    rows = read_rows(args.metrics)
    write_svg(rows, args.output, args.title)
    print(f"Question-metric SVG saved to: {args.output}")


if __name__ == "__main__":
    main()
