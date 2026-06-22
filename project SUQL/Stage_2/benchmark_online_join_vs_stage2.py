#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_stage2 import QUERIES, python_executable


STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parent
BENCH_DIR = STAGE_DIR / "benchmarks"
ONLINE_JOIN_DIR = ROOT / "src_online_join"
DEFAULT_STAGE2_RUN = BENCH_DIR / "experiment_v7_gemma2_routed_winners" / "metrics.csv"
DEFAULT_SAMPLE = BENCH_DIR / "data_sample_100" / "imdb_joined.csv"


def parse_metric(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def prepare_online_split(joined_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    structured_path = output_dir / "imdb_structured_joined.csv"
    reviews_path = output_dir / "imdb_reviews.csv"
    if structured_path.exists() and reviews_path.exists():
        return {"structured": structured_path, "reviews": reviews_path}

    joined = pd.read_csv(joined_path)
    structured_cols = ["movie_id", "title", "year", "runtime", "director", "genres"]
    joined[structured_cols].drop_duplicates("movie_id").to_csv(structured_path, index=False)
    joined[["movie_id", "review"]].to_csv(reviews_path, index=False)
    return {"structured": structured_path, "reviews": reviews_path}


def run_online_join(
    query: dict[str, str],
    run_dir: Path,
    python: str,
    split_paths: dict[str, Path],
    api_base: str,
    model: str,
) -> dict[str, object]:
    output_dir = run_dir / "outputs" / "online_join"
    log_dir = run_dir / "logs" / "online_join"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"{query['id']}.csv"
    log_path = log_dir / f"{query['id']}.log"

    env = os.environ.copy()
    env.update(
        {
            "SUQL_API_BASE": api_base,
            "SUQL_MODEL": model if model.startswith("ollama/") else f"ollama/{model}",
            "SUQL_STRUCTURED_DATA_PATH": str(split_paths["structured"]),
            "SUQL_REVIEWS_DATA_PATH": str(split_paths["reviews"]),
        }
    )
    cmd = [python, "-u", "main.py", "--suql", query["suql"], "--output", str(output_csv)]

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ONLINE_JOIN_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    log_path.write_text(proc.stdout, encoding="utf-8")

    result_rows = ""
    if output_csv.exists():
        try:
            result_rows = int(len(pd.read_csv(output_csv)))
        except Exception:
            result_rows = ""

    return {
        "project": "online_join",
        "query_id": query["id"],
        "question": query["question"],
        "exit_code": proc.returncode,
        "wall_seconds": round(wall_seconds, 2),
        "engine_seconds": parse_metric(r"Query execution time: ([0-9.]+) seconds", proc.stdout),
        "actual_engine_wall_seconds": parse_metric(r"Actual wall-clock time: ([0-9.]+) seconds", proc.stdout),
        "llm_full_calls": parse_metric(r"LLM prompts sent: (\d+)", proc.stdout),
        "llm_prompts_issued": parse_metric(r"LLM prompts sent: (\d+)", proc.stdout),
        "structured_candidates": parse_metric(r"Structural filter .*?(\d+) candidate rows", proc.stdout),
        "semantic_rows": parse_metric(r"Semantic retrieval .*?(\d+) review rows", proc.stdout),
        "join_rows": parse_metric(r"Join on movie_id .*?(\d+) rows", proc.stdout),
        "result_rows": result_rows,
        "cheap_score_calls": 0,
        "cheap_early_accept": 0,
        "cheap_early_reject": 0,
        "cheap_skipped": 0,
        "cheap_disabled": 0,
        "expensive_full_calls": parse_metric(r"LLM prompts sent: (\d+)", proc.stdout),
        "output_csv": str(output_csv.relative_to(ROOT)),
        "log_path": str(log_path.relative_to(ROOT)),
    }


def load_stage2_rows(metrics_path: Path) -> list[dict[str, object]]:
    with metrics_path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("project") == "stage2"]
    for row in rows:
        row["project"] = "stage2_v7"
        row.setdefault("actual_engine_wall_seconds", "")
    return rows


def save_metrics(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "project",
        "query_id",
        "question",
        "exit_code",
        "wall_seconds",
        "engine_seconds",
        "actual_engine_wall_seconds",
        "llm_full_calls",
        "llm_prompts_issued",
        "structured_candidates",
        "semantic_rows",
        "join_rows",
        "result_rows",
        "cheap_score_calls",
        "cheap_early_accept",
        "cheap_early_reject",
        "cheap_skipped",
        "cheap_disabled",
        "expensive_full_calls",
        "output_csv",
        "log_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    query_ids = [query["id"] for query in QUERIES]
    labels = [f"Q{i + 1}" for i in range(len(query_ids))]
    x = np.arange(len(query_ids))

    online = df[df["project"] == "online_join"].set_index("query_id").reindex(query_ids)
    stage2 = df[df["project"] == "stage2_v7"].set_index("query_id").reindex(query_ids)
    columns = [
        "wall_seconds",
        "engine_seconds",
        "llm_prompts_issued",
        "structured_candidates",
        "semantic_rows",
        "join_rows",
        "result_rows",
    ]
    for frame in (online, stage2):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    axes_flat = axes.flatten()
    fig.subplots_adjust(top=0.80, hspace=0.60, wspace=0.28)

    question_lines = ["Question Index"]
    question_lines.extend(f"Q{i + 1} - {query['question']}" for i, query in enumerate(QUERIES))
    fig.text(
        0.5,
        0.985,
        "\n".join(question_lines),
        ha="center",
        va="top",
        fontsize=10,
        fontfamily="monospace",
        fontweight="bold",
        linespacing=1.25,
    )

    def line_panel(ax, title: str, column: str, ylabel: str) -> None:
        online_y = online[column]
        stage2_y = stage2[column]
        blue = "#2b8cbe"
        red = "#f04b5f"
        ax.plot(x, online_y, marker="o", linewidth=2.5, markersize=6, color=blue, label="online_join")
        ax.plot(x, stage2_y, marker="o", linewidth=2.5, markersize=6, color=red, label="stage2_v7")
        ax.fill_between(x, online_y.fillna(0), color=blue, alpha=0.08)
        ax.fill_between(x, stage2_y.fillna(0), color=red, alpha=0.10)
        for xi, value in zip(x, online_y):
            if pd.notna(value):
                ax.annotate(f"{value:.0f}", (xi, value), textcoords="offset points", xytext=(0, 8), ha="center", color=blue, fontsize=8, fontweight="bold")
        for xi, value in zip(x, stage2_y):
            if pd.notna(value):
                ax.annotate(f"{value:.0f}", (xi, value), textcoords="offset points", xytext=(0, -12), ha="center", color=red, fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="upper right")

    panels = [
        ("Wall-clock Latency", "wall_seconds", "seconds"),
        ("Engine Time", "engine_seconds", "seconds"),
        ("LLM Prompts Issued", "llm_prompts_issued", "count"),
        ("Structured Candidates", "structured_candidates", "count"),
        ("Semantic Rows Retrieved", "semantic_rows", "count"),
        ("Join Rows", "join_rows", "count"),
        ("Result Rows", "result_rows", "count"),
    ]
    for ax, (title, column, ylabel) in zip(axes_flat, panels):
        line_panel(ax, title, column, ylabel)
    axes_flat[-1].axis("off")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare src_online_join with the latest Stage2 run.")
    parser.add_argument("--stage2-metrics", default=str(DEFAULT_STAGE2_RUN))
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--api-base", default=os.environ.get("SUQL_API_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default=os.environ.get("SUQL_MODEL", "ollama/phi4-mini"))
    parser.add_argument("--python", default=python_executable())
    parser.add_argument("--run-name", default="online_join_vs_stage2_v7")
    args = parser.parse_args()

    run_dir = BENCH_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    split_paths = prepare_online_split(Path(args.sample), run_dir / "online_split")

    rows: list[dict[str, object]] = []
    rows.extend(load_stage2_rows(Path(args.stage2_metrics)))
    for query in QUERIES:
        print(f"\n=== {query['id']} ===")
        print("Running online_join...", flush=True)
        row = run_online_join(query, run_dir, args.python, split_paths, args.api_base, args.model)
        rows.append(row)
        print(
            f"  online_join: exit={row['exit_code']} wall={row['wall_seconds']}s "
            f"prompts={row['llm_prompts_issued']} rows={row['result_rows']}"
        )

    metrics_path = run_dir / "metrics.csv"
    save_metrics(rows, metrics_path)
    plot_comparison(pd.DataFrame(rows), run_dir / "comparison_plot.png")
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Plot saved to: {run_dir / 'comparison_plot.png'}")


if __name__ == "__main__":
    main()
