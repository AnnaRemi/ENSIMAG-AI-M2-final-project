#!/usr/bin/env python3
"""
Benchmark src_baseline vs src_online_join on the same SUQL queries.

The default benchmark uses a deterministic sampled dataset so the online-join
implementation does not have to scan all reviews for every question. Use
--sample-size 0 for full data, but expect online_join to be much slower.
"""

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


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BENCH_DIR = ROOT / "benchmarks"

PROJECTS = {
    "baseline": ROOT / "src_baseline",
    "online_join": ROOT / "src_online_join",
}

DEFAULT_MODEL = os.environ.get("SUQL_MODEL", "ollama/gemma2:2b")

QUERIES = [
    {
        "id": "q1_comedies",
        "question": "Which comedy movies have reviews saying they are funny or enjoyable?",
        "suql": (
            "SELECT movie_id, title, year, runtime, director, genres, "
            "summary(review) AS review_summary FROM movies "
            "WHERE genres LIKE '%Comedy%' "
            "AND answer(review, 'Does the reviewer describe the movie as funny, humorous, amusing, or enjoyable?') = 'Yes' "
            "LIMIT 10;"
        ),
    },
    {
        "id": "q2_dramas",
        "question": "Which drama movies have emotionally strong reviews?",
        "suql": (
            "SELECT movie_id, title, year, runtime, director, genres, "
            "summary(review) AS review_summary FROM movies "
            "WHERE genres LIKE '%Drama%' "
            "AND answer(review, 'Does the reviewer describe the movie as emotionally moving, powerful, touching, or memorable?') = 'Yes' "
            "LIMIT 10;"
        ),
    },
    {
        "id": "q3_horror",
        "question": "Which horror movies have reviews saying they are scary or tense?",
        "suql": (
            "SELECT movie_id, title, year, runtime, director, genres, "
            "summary(review) AS review_summary FROM movies "
            "WHERE genres LIKE '%Horror%' "
            "AND answer(review, 'Does the reviewer describe the movie as scary, creepy, tense, frightening, or suspenseful?') = 'Yes' "
            "LIMIT 10;"
        ),
    },
    {
        "id": "q4_romance_movies",
        "question": "Which romance movies have reviews praising the love story or relationships?",
        "suql": (
            "SELECT movie_id, title, year, runtime, director, genres, "
            "summary(review) AS review_summary FROM movies "
            "WHERE genres LIKE '%Romance%' "
            "AND answer(review, 'Does the reviewer praise or positively describe the romance, love story, chemistry, or relationships?') = 'Yes' "
            "LIMIT 10;"
        ),
    },
    {
        "id": "q5_animation_movies",
        "question": "Which animated movies have reviews praising the visuals or family appeal?",
        "suql": (
            "SELECT movie_id, title, year, runtime, director, genres, "
            "summary(review) AS review_summary FROM movies "
            "WHERE genres LIKE '%Animation%' "
            "AND answer(review, 'Does the reviewer praise the animation, visuals, charm, or family-friendly appeal?') = 'Yes' "
            "LIMIT 10;"
        ),
    },
    {
        "id": "q6_action_movies",
        "question": "Which action movies have reviews saying they are exciting or thrilling?",
        "suql": (
            "SELECT movie_id, title, year, runtime, director, genres, "
            "summary(review) AS review_summary FROM movies "
            "WHERE genres LIKE '%Action%' "
            "AND answer(review, 'Does the reviewer describe the movie as exciting, thrilling, fast-paced, intense, or action-packed?') = 'Yes' "
            "LIMIT 10;"
        ),
    },
    {
        "id": "q7_scifi_movies",
        "question": "Which science fiction movies have reviews praising imagination, ideas, or effects?",
        "suql": (
            "SELECT movie_id, title, year, runtime, director, genres, "
            "summary(review) AS review_summary FROM movies "
            "WHERE genres LIKE '%Sci-Fi%' "
            "AND answer(review, 'Does the reviewer praise the science fiction ideas, imagination, world-building, visuals, or special effects?') = 'Yes' "
            "LIMIT 10;"
        ),
    },
    {
        "id": "q8_movies_1990s",
        "question": "Which movies from the 1990s have reviews recommending the movie?",
        "suql": (
            "SELECT movie_id, title, year, runtime, director, genres, "
            "summary(review) AS review_summary FROM movies "
            "WHERE year >= 1990 AND year <= 1999 "
            "AND answer(review, 'Does the reviewer recommend the movie or express an overall positive opinion?') = 'Yes' "
            "LIMIT 10;"
        ),
    },
    {
        "id": "q9_movies_2000s",
        "question": "Which movies from 2000 onward have reviews criticizing the movie?",
        "suql": (
            "SELECT movie_id, title, year, runtime, director, genres, "
            "summary(review) AS review_summary FROM movies "
            "WHERE year >= 2000 "
            "AND answer(review, 'Does the reviewer criticize, dislike, or express disappointment with the movie?') = 'Yes' "
            "LIMIT 10;"
        ),
    },
]


def _python_executable() -> str:
    baseline_venv_python = PROJECTS["baseline"] / ".venv" / "bin" / "python"
    if baseline_venv_python.exists():
        return str(baseline_venv_python)
    return sys.executable


def litellm_model_name(model: str) -> str:
    return model if model.startswith("ollama/") else f"ollama/{model}"


def _validate_joined_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing joined data file: {path}")

    columns = set(pd.read_csv(path, nrows=0).columns)
    required = {"movie_id", "title", "year", "runtime", "director", "genres", "review"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _split_joined_sample(joined_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    structured_path = output_dir / "imdb_structured_joined.csv"
    reviews_path = output_dir / "imdb_reviews.csv"

    if not structured_path.exists() or not reviews_path.exists():
        joined = pd.read_csv(joined_path)
        structured_cols = ["movie_id", "title", "year", "runtime", "director", "genres"]
        joined[structured_cols].drop_duplicates("movie_id").to_csv(structured_path, index=False)
        joined[["movie_id", "review"]].to_csv(reviews_path, index=False)

    return {
        "baseline_joined": joined_path,
        "online_structured": structured_path,
        "online_reviews": reviews_path,
    }


def prepare_explicit_sample(
    data_path: Path | None,
    data_sample_dir: Path | None,
) -> dict[str, Path] | None:
    if data_path and data_sample_dir:
        raise ValueError("Use either --data-path or --data-sample-dir, not both.")

    if data_path:
        data_path = data_path.resolve()
        _validate_joined_path(data_path)
        return _split_joined_sample(data_path, data_path.parent)

    if not data_sample_dir:
        return None

    sample_dir = data_sample_dir.resolve()
    joined_path = sample_dir / "imdb_joined.csv"
    structured_path = sample_dir / "imdb_structured_joined.csv"
    reviews_path = sample_dir / "imdb_reviews.csv"
    _validate_joined_path(joined_path)

    if not structured_path.exists() or not reviews_path.exists():
        return _split_joined_sample(joined_path, sample_dir)

    return {
        "baseline_joined": joined_path,
        "online_structured": structured_path,
        "online_reviews": reviews_path,
    }


def prepare_sample(sample_size: int, seed: int) -> dict[str, Path]:
    BENCH_DIR.mkdir(exist_ok=True)
    sample_dir = BENCH_DIR / f"data_sample_{sample_size}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if sample_size <= 0:
        return {
            "baseline_joined": DATA_DIR / "imdb_joined.csv",
            "online_structured": DATA_DIR / "imdb_structured_joined.csv",
            "online_reviews": DATA_DIR / "imdb_reviews.csv",
        }

    joined_path = sample_dir / "imdb_joined.csv"
    structured_path = sample_dir / "imdb_structured_joined.csv"
    reviews_path = sample_dir / "imdb_reviews.csv"

    if joined_path.exists() and structured_path.exists() and reviews_path.exists():
        return {
            "baseline_joined": joined_path,
            "online_structured": structured_path,
            "online_reviews": reviews_path,
        }

    joined = pd.read_csv(DATA_DIR / "imdb_joined.csv")
    sample_n = min(sample_size, len(joined))
    sampled = joined.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    sampled.to_csv(joined_path, index=False)

    structured_cols = ["movie_id", "title", "year", "runtime", "director", "genres"]
    sampled[structured_cols].drop_duplicates("movie_id").to_csv(structured_path, index=False)
    sampled[["movie_id", "review"]].to_csv(reviews_path, index=False)

    return {
        "baseline_joined": joined_path,
        "online_structured": structured_path,
        "online_reviews": reviews_path,
    }


def parse_metric(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def run_project(
    project_name: str,
    query: dict[str, str],
    python: str,
    data_paths: dict[str, Path],
    api_base: str,
    model: str,
    run_dir: Path,
) -> dict[str, str | float]:
    project_dir = PROJECTS[project_name]
    output_dir = run_dir / "outputs" / project_name
    log_dir = run_dir / "logs" / project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"{query['id']}.csv"
    log_path = log_dir / f"{query['id']}.log"

    env = os.environ.copy()
    env["SUQL_API_BASE"] = api_base
    env["SUQL_MODEL"] = model
    if project_name == "baseline":
        env["SUQL_DATA_PATH"] = str(data_paths["baseline_joined"])
    else:
        env["SUQL_STRUCTURED_DATA_PATH"] = str(data_paths["online_structured"])
        env["SUQL_REVIEWS_DATA_PATH"] = str(data_paths["online_reviews"])

    cmd = [
        python,
        "-u",
        "main.py",
        "--suql",
        query["suql"],
        "--output",
        str(output_csv),
    ]

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

    row_count = ""
    if output_csv.exists():
        try:
            row_count = str(len(pd.read_csv(output_csv)))
        except Exception:
            row_count = ""

    return {
        "project": project_name,
        "query_id": query["id"],
        "question": query["question"],
        "exit_code": proc.returncode,
        "wall_seconds": round(wall_seconds, 2),
        "engine_seconds": parse_metric(r"Query execution time: ([0-9.]+) seconds", proc.stdout),
        "llm_prompts": parse_metric(r"LLM prompts sent: (\d+)", proc.stdout),
        "structured_candidates": parse_metric(r"Structural filter .*?(\d+) candidate rows", proc.stdout),
        "semantic_rows": parse_metric(r"Semantic retrieval .*?(\d+) review rows", proc.stdout),
        "join_rows": parse_metric(r"Join on movie_id .*?(\d+) rows", proc.stdout),
        "result_rows": row_count,
        "output_csv": str(output_csv.relative_to(ROOT)),
        "log_path": str(log_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs online_join.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Rows sampled from imdb_joined.csv. Use 0 for full data.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Explicit imdb_joined.csv to benchmark. Overrides --sample-size sampling.",
    )
    parser.add_argument(
        "--data-sample-dir",
        type=Path,
        help="Directory containing imdb_joined.csv and optional online split CSVs. Overrides --sample-size sampling.",
    )
    parser.add_argument("--api-base", default=os.environ.get("SUQL_API_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=_python_executable())
    parser.add_argument("--run-name", help="Benchmark output folder name under benchmarks/.")
    args = parser.parse_args()
    args.model = litellm_model_name(args.model)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = BENCH_DIR / (args.run_name or f"run_{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    data_paths = prepare_explicit_sample(args.data_path, args.data_sample_dir)
    if data_paths is None:
        data_paths = prepare_sample(args.sample_size, args.seed)
    metrics_path = run_dir / "metrics.csv"

    rows = []
    for query in QUERIES:
        print(f"\n=== {query['id']} ===")
        print(query["question"])
        for project_name in ("baseline", "online_join"):
            print(f"Running {project_name}...", flush=True)
            row = run_project(
                project_name=project_name,
                query=query,
                python=args.python,
                data_paths=data_paths,
                api_base=args.api_base,
                model=args.model,
                run_dir=run_dir,
            )
            rows.append(row)
            print(
                f"  {project_name}: exit={row['exit_code']} "
                f"wall={row['wall_seconds']}s prompts={row['llm_prompts']} rows={row['result_rows']}"
            )

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
