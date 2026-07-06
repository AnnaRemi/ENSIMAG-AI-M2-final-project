#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parent
DATA_DIR = ROOT / "data"
BENCH_DIR = STAGE_DIR / "benchmarks"
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plot_benchmarks import plot_per_question_metrics

PROJECTS = {
    "baseline": ROOT / "src_baseline",
    "stage1": ROOT / "src_baseline_stage1",
}

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
            "SUQL_MODEL": model,
            "SUQL_DATA_PATH": str(data_path),
            "SUQL_METRICS_PATH": str(sidecar_path),
            "SUQL_THRESHOLDS_PATH": str(thresholds_path),
        }
    )
    cmd = [
        python,
        "-u",
        str(ROOT / "scripts" / "run_suql.py"),
        "--engine-dir",
        str(project_dir),
        "--suql",
        query["suql"],
        "--output",
        str(output_csv),
    ]

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
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
        "structured_candidates": sidecar.get("structured_candidates", ""),
        "semantic_rows": sidecar.get("semantic_rows", ""),
        "join_rows": sidecar.get("join_rows", ""),
        "result_rows": sidecar.get("result_rows", ""),
        "output_csv": str(output_csv.relative_to(ROOT)),
        "log_path": str(log_path.relative_to(ROOT)),
        "metrics_sidecar": str(sidecar_path.relative_to(ROOT)),
    }


def save_metrics(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs Stage_1 threshold runtime.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Explicit imdb_joined.csv to benchmark. Overrides --sample-size sampling.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--api-base", default=os.environ.get("SUQL_API_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default=os.environ.get("SUQL_MODEL", "ollama/phi4-mini"))
    parser.add_argument("--thresholds", default=str(STAGE_DIR / "thresholds.json"))
    parser.add_argument("--python", default=python_executable())
    parser.add_argument("--run-name")
    args = parser.parse_args()

    run_name = args.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = BENCH_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path if args.data_path else prepare_sample(args.sample_size, args.seed)
    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark data file does not exist: {data_path}")
    thresholds_path = Path(args.thresholds)

    rows = []
    for query in QUERIES:
        print(f"\n=== {query['id']} ===")
        for project in ("baseline", "stage1"):
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
            )
            rows.append(row)
            print(
                f"  {project}: exit={row['exit_code']} wall={row['wall_seconds']}s "
                f"full={row['llm_full_calls']} accept={row['llm_early_accept']} "
                f"reject={row['llm_early_reject']} rows={row['result_rows']}"
            )

    metrics_path = run_dir / "metrics.csv"
    save_metrics(rows, metrics_path)
    plot_per_question_metrics(
        metrics_csv=metrics_path,
        output=run_dir / "comparison_plot.png",
        implementations=["baseline", "stage1"],
        title="Baseline vs Stage 1",
    )
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Plot saved to: {run_dir / 'comparison_plot.png'}")


if __name__ == "__main__":
    main()
