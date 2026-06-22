#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parent
DATA_DIR = ROOT / "data"
BENCH_DIR = STAGE_DIR / "benchmarks"

PROJECTS = {
    "baseline": ROOT / "src_baseline",
    "stage2": ROOT / "src_baseline_stage2",
}

DEFAULT_EXPENSIVE_MODEL = os.environ.get("SUQL_EXPENSIVE_MODEL", os.environ.get("SUQL_MODEL", "ollama/phi4-mini"))
DEFAULT_CHEAP_MODEL = os.environ.get("SUQL_CHEAP_MODEL", "ollama/gemma2:2b")
DEFAULT_CHEAP_ACCEPT_FLOOR = float(os.environ.get("SUQL_CHEAP_ACCEPT_FLOOR", "4.0"))
DEFAULT_CHEAP_MIN_DECISION_RATE = float(os.environ.get("SUQL_CHEAP_MIN_DECISION_RATE", "0.3"))
DEFAULT_CHEAP_MIN_PROBES = int(os.environ.get("SUQL_CHEAP_MIN_PROBES", "5"))


def litellm_model_name(model: str) -> str:
    return model if model.startswith("ollama/") else f"ollama/{model}"


def answer_question(query: dict[str, str]) -> str:
    match = re.search(r"answer\s*\(\s*review\s*,\s*(['\"])(.*?)\1\s*\)", query["suql"], re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(2)


def model_usage_summary(sidecar: dict) -> str:
    usage_by_question = sidecar.get("model_usage_by_question") or {}
    if not usage_by_question:
        return ""

    parts = []
    for question, usage in usage_by_question.items():
        parts.append(
            "{question} | cheap={cheap_model} score={cheap_score_calls} "
            "fail={cheap_score_failures} accept={cheap_early_accept} "
            "reject={cheap_early_reject} skipped={cheap_skipped} disabled={cheap_disabled} "
            "| expensive={expensive_model} "
            "full={expensive_full_calls} | accept_floor={accept_floor} "
            "| effective_accept={effective_accept} | min_rate={min_rate} "
            "| min_probes={min_probes} | skip_reason={skip_reason} "
            "| failure_reasons={failure_reasons}".format(
                question=str(question).replace("\n", " "),
                cheap_model=usage.get("cheap_model", ""),
                cheap_score_calls=usage.get("cheap_score_calls", 0),
                cheap_score_failures=usage.get("cheap_score_failures", 0),
                cheap_early_accept=usage.get("cheap_early_accept", 0),
                cheap_early_reject=usage.get("cheap_early_reject", 0),
                cheap_skipped=usage.get("cheap_skipped", 0),
                cheap_disabled=usage.get("cheap_disabled", 0),
                expensive_model=usage.get("expensive_model", ""),
                expensive_full_calls=usage.get("expensive_full_calls", 0),
                accept_floor=usage.get("cheap_accept_floor", ""),
                effective_accept=usage.get("effective_accept_threshold", ""),
                min_rate=usage.get("cheap_min_decision_rate", ""),
                min_probes=usage.get("cheap_min_probes", ""),
                skip_reason=usage.get("cheap_skip_reason", ""),
                failure_reasons=json.dumps(usage.get("cheap_score_failure_reasons", {}), sort_keys=True),
            )
        )
    return " || ".join(parts)

QUERIES = [
    {
        "id": "q1_comedy_1990s_funny",
        "question": "Which comedy movies from the 1990s have reviews saying they are funny?",
        "suql": "SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary FROM movies WHERE year >= 1990 AND year <= 1999 AND genres LIKE '%Comedy%' AND answer(review, 'Does the reviewer describe the movie as funny, humorous, witty, or entertaining?') = 'Yes' LIMIT 10;",
    },
    {
        "id": "q2_drama_2000s_acting",
        "question": "Which drama movies from 2000 onward have reviews praising the acting?",
        "suql": "SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary FROM movies WHERE year >= 2000 AND genres LIKE '%Drama%' AND answer(review, 'Does the reviewer praise the acting, performances, cast, or characters?') = 'Yes' LIMIT 10;",
    },
    {
        "id": "q3_horror_short_suspense",
        "question": "Which horror movies under 110 minutes have reviews mentioning suspense or tension?",
        "suql": "SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary FROM movies WHERE genres LIKE '%Horror%' AND runtime < 110 AND answer(review, 'Does the reviewer mention suspense, tension, scares, fear, or creepy atmosphere?') = 'Yes' LIMIT 10;",
    },
    {
        "id": "q4_romance_1990s_chemistry",
        "question": "Which romance movies from the 1990s have reviews praising chemistry or relationships?",
        "suql": "SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary FROM movies WHERE year >= 1990 AND year <= 1999 AND genres LIKE '%Romance%' AND answer(review, 'Does the reviewer praise the chemistry, romance, love story, or relationships?') = 'Yes' LIMIT 10;",
    },
    {
        "id": "q5_action_short_exciting",
        "question": "Which action movies under 120 minutes have reviews saying they are exciting?",
        "suql": "SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary FROM movies WHERE genres LIKE '%Action%' AND runtime < 120 AND answer(review, 'Does the reviewer describe the movie as exciting, thrilling, intense, energetic, or action-packed?') = 'Yes' LIMIT 10;",
    },
    {
        "id": "q6_scifi_1980plus_ideas",
        "question": "Which science fiction movies from 1980 onward have reviews praising ideas or effects?",
        "suql": "SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary FROM movies WHERE year >= 1980 AND genres LIKE '%Sci-Fi%' AND answer(review, 'Does the reviewer praise the science fiction ideas, imagination, visuals, or special effects?') = 'Yes' LIMIT 10;",
    },
    {
        "id": "q7_adventure_1990plus_family",
        "question": "Which adventure movies from 1990 onward have reviews praising family appeal or charm?",
        "suql": "SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary FROM movies WHERE year >= 1990 AND genres LIKE '%Adventure%' AND answer(review, 'Does the reviewer praise the movie as charming, family-friendly, fun, or adventurous?') = 'Yes' LIMIT 10;",
    },
    {
        "id": "q8_crime_1990plus_writing",
        "question": "Which crime movies from 1990 onward have reviews praising writing or plot?",
        "suql": "SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary FROM movies WHERE year >= 1990 AND genres LIKE '%Crime%' AND answer(review, 'Does the reviewer praise the writing, script, story, plot, or dialogue?') = 'Yes' LIMIT 10;",
    },
    {
        "id": "q9_old_movies_criticism",
        "question": "Which movies before 1980 have reviews criticizing pacing or quality?",
        "suql": "SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary FROM movies WHERE year < 1980 AND answer(review, 'Does the reviewer criticize the movie, its pacing, quality, acting, story, or production?') = 'Yes' LIMIT 10;",
    },
]


def python_executable() -> str:
    baseline_venv_python = ROOT / "src_baseline" / ".venv" / "bin" / "python"
    return str(baseline_venv_python) if baseline_venv_python.exists() else sys.executable


def prepare_sample(sample_size: int, seed: int) -> Path:
    if sample_size <= 0:
        return DATA_DIR / "imdb_joined.csv"
    sample_dir = BENCH_DIR / f"data_sample_{sample_size}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "imdb_joined.csv"
    if not sample_path.exists():
        df = pd.read_csv(DATA_DIR / "imdb_joined.csv")
        df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True).to_csv(sample_path, index=False)
    return sample_path


def run_project(
    project: str,
    query: dict[str, str],
    run_dir: Path,
    python: str,
    data_path: Path,
    thresholds_path: Path,
    api_base: str,
    model: str,
    cheap_model: str,
    cheap_accept_floor: float,
    cheap_min_decision_rate: float,
    cheap_min_probes: int,
    cheap_disabled_questions: list[str],
) -> dict:
    project_dir = PROJECTS[project]
    output_dir = run_dir / "outputs" / project
    log_dir = run_dir / "logs" / project
    sidecar_dir = run_dir / "metrics_sidecars" / project
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"{query['id']}.csv"
    log_path = log_dir / f"{query['id']}.log"
    sidecar_path = sidecar_dir / f"{query['id']}.json"

    env = os.environ.copy()
    env.update(
        {
            "SUQL_API_BASE": api_base,
            "SUQL_MODEL": litellm_model_name(model),
            "SUQL_EXPENSIVE_MODEL": litellm_model_name(model),
            "SUQL_CHEAP_MODEL": litellm_model_name(cheap_model),
            "SUQL_CHEAP_ACCEPT_FLOOR": str(float(cheap_accept_floor)),
            "SUQL_CHEAP_MIN_DECISION_RATE": str(float(cheap_min_decision_rate)),
            "SUQL_CHEAP_MIN_PROBES": str(int(cheap_min_probes)),
            "SUQL_CHEAP_DISABLED_QUESTIONS": json.dumps(cheap_disabled_questions),
            "SUQL_DATA_PATH": str(data_path),
            "SUQL_METRICS_PATH": str(sidecar_path),
            "SUQL_THRESHOLDS_PATH": str(thresholds_path),
        }
    )
    cmd = [python, "-u", "main.py", "--suql", query["suql"], "--output", str(output_csv)]

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=project_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    log_path.write_text(proc.stdout, encoding="utf-8")

    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    else:
        sidecar = {}

    return {
        "project": project,
        "query_id": query["id"],
        "question": query["question"],
        "exit_code": proc.returncode,
        "wall_seconds": round(wall_seconds, 2),
        "engine_seconds": sidecar.get("engine_seconds", ""),
        "llm_full_calls": sidecar.get("llm_full_calls", ""),
        "llm_prompts_issued": sidecar.get("llm_prompts_issued", sidecar.get("llm_full_calls", "")),
        "llm_early_accept": sidecar.get("llm_early_accept", 0),
        "llm_early_reject": sidecar.get("llm_early_reject", 0),
        "cheap_model": sidecar.get("cheap_model", litellm_model_name(cheap_model) if project == "stage2" else ""),
        "expensive_model": sidecar.get("expensive_model", litellm_model_name(model)),
        "cheap_accept_floor": sidecar.get("cheap_accept_floor", cheap_accept_floor if project == "stage2" else ""),
        "cheap_min_decision_rate": sidecar.get(
            "cheap_min_decision_rate", cheap_min_decision_rate if project == "stage2" else ""
        ),
        "cheap_min_probes": sidecar.get("cheap_min_probes", cheap_min_probes if project == "stage2" else ""),
        "cheap_disabled_questions": json.dumps(sidecar.get("cheap_disabled_questions", []), sort_keys=True),
        "cheap_score_failure_reasons": json.dumps(sidecar.get("cheap_score_failure_reasons", {}), sort_keys=True),
        "model_usage_summary": model_usage_summary(sidecar),
        "structured_candidates": sidecar.get("structured_candidates", ""),
        "semantic_rows": sidecar.get("semantic_rows", ""),
        "join_rows": sidecar.get("join_rows", ""),
        "result_rows": sidecar.get("result_rows", ""),
        "cheap_score_calls": sidecar.get("cheap_score_calls", 0),
        "cheap_score_failures": sidecar.get("cheap_score_failures", 0),
        "expensive_full_calls": sidecar.get("expensive_full_calls", sidecar.get("llm_full_calls", "")),
        "cheap_early_accept": sidecar.get("cheap_early_accept", sidecar.get("llm_early_accept", 0)),
        "cheap_early_reject": sidecar.get("cheap_early_reject", sidecar.get("llm_early_reject", 0)),
        "cheap_skipped": sidecar.get("cheap_skipped", 0),
        "cheap_disabled": sidecar.get("cheap_disabled", 0),
        "output_csv": str(output_csv.relative_to(ROOT)),
        "log_path": str(log_path.relative_to(ROOT)),
        "metrics_sidecar": str(sidecar_path.relative_to(ROOT)),
    }


def save_metrics(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    query_ids = [q["id"] for q in QUERIES]
    labels = [f"Q{i + 1}" for i in range(len(query_ids))]
    x = np.arange(len(query_ids))
    fallback_columns = {
        "engine_seconds": "wall_seconds",
        "llm_prompts_issued": "llm_full_calls",
        "structured_candidates": "result_rows",
        "semantic_rows": "result_rows",
        "join_rows": "result_rows",
    }
    for column, fallback in fallback_columns.items():
        if column not in df.columns:
            df[column] = df.get(fallback)

    baseline = df[df["project"] == "baseline"].set_index("query_id").reindex(query_ids)
    stage2 = df[df["project"] == "stage2"].set_index("query_id").reindex(query_ids)
    for frame in (baseline, stage2):
        for column in [
            "wall_seconds",
            "engine_seconds",
            "llm_full_calls",
            "llm_prompts_issued",
            "llm_early_accept",
            "llm_early_reject",
            "structured_candidates",
            "semantic_rows",
            "join_rows",
            "result_rows",
        ]:
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
        base_y = baseline[column]
        stage_y = stage2[column]
        blue = "#2b8cbe"
        red = "#f04b5f"

        ax.plot(x, base_y, marker="o", linewidth=2.5, markersize=6, color=blue, label="baseline")
        ax.plot(x, stage_y, marker="o", linewidth=2.5, markersize=6, color=red, label="stage2")
        ax.fill_between(x, base_y.fillna(0), color=blue, alpha=0.08)
        ax.fill_between(x, stage_y.fillna(0), color=red, alpha=0.10)

        for xi, value in zip(x, base_y):
            if pd.notna(value):
                ax.annotate(
                    f"{value:.0f}",
                    (xi, value),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    color=blue,
                    fontsize=8,
                    fontweight="bold",
                )
        for xi, value in zip(x, stage_y):
            if pd.notna(value):
                ax.annotate(
                    f"{value:.0f}",
                    (xi, value),
                    textcoords="offset points",
                    xytext=(0, -12),
                    ha="center",
                    color=red,
                    fontsize=8,
                    fontweight="bold",
                )

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
    parser = argparse.ArgumentParser(description="Benchmark baseline vs Stage_2 cheap-to-expensive cascade runtime.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--api-base", default=os.environ.get("SUQL_API_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default=DEFAULT_EXPENSIVE_MODEL)
    parser.add_argument("--cheap-model", default=DEFAULT_CHEAP_MODEL)
    parser.add_argument("--cheap-accept-floor", type=float, default=DEFAULT_CHEAP_ACCEPT_FLOOR)
    parser.add_argument("--cheap-min-decision-rate", type=float, default=DEFAULT_CHEAP_MIN_DECISION_RATE)
    parser.add_argument("--cheap-min-probes", type=int, default=DEFAULT_CHEAP_MIN_PROBES)
    parser.add_argument(
        "--cheap-disabled-query-ids",
        nargs="*",
        default=[],
        help="Benchmark query IDs whose answer() question should bypass the cheap model.",
    )
    parser.add_argument(
        "--cheap-disabled-questions",
        nargs="*",
        default=[],
        help="Exact answer() question strings that should bypass the cheap model.",
    )
    parser.add_argument("--thresholds", default=str(STAGE_DIR / "thresholds.json"))
    parser.add_argument("--python", default=python_executable())
    parser.add_argument("--run-name")
    args = parser.parse_args()
    args.model = litellm_model_name(args.model)
    args.cheap_model = litellm_model_name(args.cheap_model)

    run_name = args.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = BENCH_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = prepare_sample(args.sample_size, args.seed)
    thresholds_path = Path(args.thresholds)
    disabled_ids = set(args.cheap_disabled_query_ids)
    cheap_disabled_questions = list(args.cheap_disabled_questions)
    for query in QUERIES:
        if query["id"] in disabled_ids:
            question = answer_question(query)
            if question:
                cheap_disabled_questions.append(question)

    rows = []
    for query in QUERIES:
        print(f"\n=== {query['id']} ===")
        for project in ("baseline", "stage2"):
            print(f"Running {project}...", flush=True)
            row = run_project(
                project=project,
                query=query,
                run_dir=run_dir,
                python=args.python,
                data_path=data_path,
                thresholds_path=thresholds_path,
                api_base=args.api_base,
                model=args.model,
                cheap_model=args.cheap_model,
                cheap_accept_floor=args.cheap_accept_floor,
                cheap_min_decision_rate=args.cheap_min_decision_rate,
                cheap_min_probes=args.cheap_min_probes,
                cheap_disabled_questions=cheap_disabled_questions,
            )
            rows.append(row)
            print(
                f"  {project}: exit={row['exit_code']} wall={row['wall_seconds']}s "
                f"full={row['llm_full_calls']} accept={row['llm_early_accept']} "
                f"reject={row['llm_early_reject']} rows={row['result_rows']}"
            )

    metrics_path = run_dir / "metrics.csv"
    save_metrics(rows, metrics_path)
    plot_comparison(pd.DataFrame(rows), run_dir / "comparison_plot.png")
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Plot saved to: {run_dir / 'comparison_plot.png'}")


if __name__ == "__main__":
    main()
