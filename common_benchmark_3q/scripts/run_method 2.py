#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from common import (
    LAB_ROOT,
    benchmark,
    cpu_seconds,
    load_movies,
    load_reviews,
    question_dir,
    write_csv,
)


def run_suql(args: argparse.Namespace, spec: dict, output_dir: Path) -> dict:
    engine_dir = LAB_ROOT / "project SUQL" / "src_baseline"
    metrics_path = output_dir / "engine_metrics.json"
    os.environ["SUQL_DATA_PATH"] = str(
        question_dir(args.question_dir) / "data" / "imdb_joined.csv"
    )
    os.environ["SUQL_API_BASE"] = args.api_base
    os.environ["SUQL_MODEL"] = args.expensive_model
    os.environ["SUQL_METRICS_PATH"] = str(metrics_path)
    sys.path.insert(0, str(engine_dir))
    import suql_engine

    suql_engine._answer_cache.clear()
    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    results = suql_engine.ask_with_suql(
        spec["suql_query"],
        output_csv=str(output_dir / "found_rows.csv"),
        verbose=True,
    )
    elapsed = time.perf_counter() - started_wall
    engine_metrics = json.loads(metrics_path.read_text())
    return {
        "implementation": "suql_baseline",
        "mode": "llm",
        "model": args.expensive_model,
        "cheap_model": "",
        "expensive_model": args.expensive_model,
        "cpu_seconds": cpu_seconds() - started_cpu,
        "engine_seconds": float(engine_metrics["engine_seconds"]),
        "wall_seconds": elapsed,
        "llm_calls": int(engine_metrics["llm_prompts_issued"]),
        "block_join_calls": 0,
        "cheap_calls": 0,
        "expensive_calls": int(engine_metrics["llm_prompts_issued"]),
        "final_answer_rows": int(len(results)),
        "found_movie_ids": sorted(results["movie_id"].astype(str).unique()),
        "structured_candidates": int(engine_metrics["structured_candidates"]),
    }


def prune_inputs(args: argparse.Namespace, spec: dict):
    v3_root = LAB_ROOT / "project Trummer" / "heterogen_v3"
    sys.path.insert(0, str(v3_root))
    from trummer_join.structured_filter import (
        apply_structured_filters,
        extract_structured_filters,
    )

    input_movies = load_movies(args.question_dir)
    input_reviews = load_reviews(args.question_dir)
    movies_frame = pd.DataFrame(input_movies)
    reviews_frame = pd.DataFrame(input_reviews)
    filters = extract_structured_filters(spec["question"], movies_frame.columns)
    movies = apply_structured_filters(movies_frame, filters).reset_index(drop=True)
    movie_ids = set(movies["movie_id"].astype(str))
    reviews = pd.DataFrame(
        [
            row
            for row in input_reviews
            if str(row.get("tconst", "")) in movie_ids
        ],
        columns=reviews_frame.columns,
    ).reset_index(drop=True)
    return input_movies, input_reviews, movies, reviews, filters


def run_v2_2(args: argparse.Namespace, spec: dict, output_dir: Path) -> dict:
    # Import structured filtering before switching to the v2_2 package.
    input_movies, input_reviews, movies, reviews, filters = prune_inputs(args, spec)
    for name in list(sys.modules):
        if name == "trummer_join" or name.startswith("trummer_join."):
            del sys.modules[name]
    v2_2_root = LAB_ROOT / "project Trummer" / "heterogen_v2_2"
    sys.path.insert(0, str(v2_2_root))
    from trummer_join.client import ChatClient
    from trummer_join.operators import block_join

    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    stats, joined = block_join(
        ChatClient(api_base=args.api_base, timeout=args.request_timeout),
        movies,
        reviews,
        spec["semantic_question"],
        args.expensive_model,
        selectivity_estimate=0.05,
        token_threshold=args.token_threshold,
        max_completion_tokens=args.max_completion_tokens,
        max_block_1_size=args.max_movie_block_size,
        max_block_2_size=args.max_review_block_size,
    )
    elapsed = time.perf_counter() - started_wall
    joined_rows = joined.to_dict("records") if not joined.empty else []
    final_rows = []
    seen = set()
    for row in joined_rows:
        movie_id = str(row.get("movie_id", ""))
        if movie_id not in seen:
            seen.add(movie_id)
            final_rows.append(
                {
                    key: row.get(key, "")
                    for key in (
                        "movie_id",
                        "title",
                        "year",
                        "runtime",
                        "director",
                        "genres",
                    )
                }
            )
    stats.to_csv(output_dir / "join_stats.csv", index=False)
    write_csv(output_dir / "joined_evidence.csv", joined_rows)
    write_csv(output_dir / "found_rows.csv", final_rows)
    return {
        "implementation": "trummer_heterogen_v2_2_structured_pruned",
        "mode": "llm",
        "model": args.expensive_model,
        "cheap_model": "",
        "expensive_model": args.expensive_model,
        "cpu_seconds": cpu_seconds() - started_cpu,
        "engine_seconds": elapsed,
        "wall_seconds": elapsed,
        "llm_calls": int(len(stats)),
        "block_join_calls": int(len(stats)),
        "cheap_calls": 0,
        "expensive_calls": int(len(stats)),
        "final_answer_rows": len(final_rows),
        "found_movie_ids": sorted(seen),
        "original_movies": len(input_movies),
        "original_reviews": len(input_reviews),
        "pruned_movies": len(movies),
        "pruned_reviews": len(reviews),
        "structured_filters": [item.as_dict() for item in filters],
        "prompt_tokens": int(stats["tokens_read"].sum()) if "tokens_read" in stats else 0,
        "completion_tokens": int(stats["tokens_written"].sum()) if "tokens_written" in stats else 0,
    }


def run_v3(args: argparse.Namespace, spec: dict, output_dir: Path) -> dict:
    v3_root = LAB_ROOT / "project Trummer" / "heterogen_v3"
    sys.path.insert(0, str(v3_root))
    from trummer_join.cascade import CascadeConfig, CascadeJoin, metrics_dict
    from trummer_join.structured_filter import (
        apply_structured_filters,
        extract_structured_filters,
    )

    input_movies = load_movies(args.question_dir)
    input_reviews = load_reviews(args.question_dir)
    movies_frame = pd.DataFrame(input_movies)
    reviews_frame = pd.DataFrame(input_reviews)
    filters = extract_structured_filters(spec["question"], movies_frame.columns)
    pruned_movies = apply_structured_filters(movies_frame, filters).reset_index(drop=True)
    movie_ids = set(pruned_movies["movie_id"].astype(str))
    pruned_reviews = pd.DataFrame(
        [row for row in input_reviews if str(row.get("tconst", "")) in movie_ids],
        columns=reviews_frame.columns,
    ).reset_index(drop=True)
    movies = pruned_movies.to_dict("records")
    reviews = pruned_reviews.to_dict("records")
    config = CascadeConfig(
        api_base=args.api_base,
        cheap_model=args.cheap_model,
        expensive_model=args.expensive_model,
        accept_threshold=args.cheap_accept_threshold,
        reject_threshold=args.cheap_reject_threshold,
        expensive_batch_size=args.expensive_batch_size,
        max_expensive_calls=args.max_expensive_calls,
        request_timeout=args.request_timeout,
    )
    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    rows, decisions, metrics = CascadeJoin(config).run(
        movies,
        reviews,
        spec["semantic_question"],
    )
    elapsed = time.perf_counter() - started_wall
    final_rows = []
    seen = set()
    for row in rows:
        movie_id = str(row.get("movie_id", ""))
        if movie_id not in seen:
            seen.add(movie_id)
            final_rows.append(
                {
                    key: row.get(key, "")
                    for key in (
                        "movie_id",
                        "title",
                        "year",
                        "runtime",
                        "director",
                        "genres",
                        "match_source",
                    )
                }
            )
    write_csv(output_dir / "cascade_decisions.csv", [asdict(item) for item in decisions])
    write_csv(output_dir / "joined_evidence.csv", rows)
    write_csv(output_dir / "found_rows.csv", final_rows)
    return {
        "implementation": "trummer_heterogen_v3_pruned_cascade",
        "mode": "llm",
        "model": f"{args.cheap_model}->{args.expensive_model}",
        "cheap_model": args.cheap_model,
        "expensive_model": args.expensive_model,
        **metrics_dict(metrics),
        "cpu_seconds": cpu_seconds() - started_cpu,
        "engine_seconds": elapsed,
        "wall_seconds": elapsed,
        "llm_calls": metrics.cheap_calls + metrics.expensive_calls,
        "block_join_calls": metrics.expensive_calls,
        "final_answer_rows": len(final_rows),
        "found_movie_ids": sorted(seen),
        "original_movies": len(input_movies),
        "original_reviews": len(input_reviews),
        "pruned_movies": len(movies),
        "pruned_reviews": len(reviews),
        "structured_filters": [item.as_dict() for item in filters],
        "max_expensive_calls": args.max_expensive_calls,
    }


def run_v2_3(args: argparse.Namespace, spec: dict, output_dir: Path) -> dict:
    input_movies, input_reviews, movies_frame, reviews_frame, filters = (
        prune_inputs(args, spec)
    )
    for name in list(sys.modules):
        if name == "trummer_join" or name.startswith("trummer_join."):
            del sys.modules[name]
    v2_3_root = LAB_ROOT / "project Trummer" / "heterogen_v2_3"
    sys.path.insert(0, str(v2_3_root))
    from trummer_join.cascade import CascadeConfig, CascadeJoin, metrics_dict

    movies = movies_frame.to_dict("records")
    reviews = reviews_frame.to_dict("records")
    config = CascadeConfig(
        api_base=args.api_base,
        cheap_model=args.cheap_model,
        expensive_model=args.expensive_model,
        accept_threshold=args.cheap_accept_threshold,
        reject_threshold=args.cheap_reject_threshold,
        cheap_batch_size=args.cheap_batch_size,
        expensive_batch_size=args.v2_3_expensive_batch_size,
        request_timeout=args.request_timeout,
    )
    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    rows, decisions, metrics = CascadeJoin(config).run(
        movies,
        reviews,
        spec["semantic_question"],
    )
    elapsed = time.perf_counter() - started_wall
    final_rows = []
    seen = set()
    for row in rows:
        movie_id = str(row.get("movie_id", ""))
        if movie_id in seen:
            continue
        seen.add(movie_id)
        final_rows.append(
            {
                key: row.get(key, "")
                for key in (
                    "movie_id",
                    "title",
                    "year",
                    "runtime",
                    "director",
                    "genres",
                    "match_source",
                )
            }
        )
    write_csv(
        output_dir / "cascade_decisions.csv",
        [asdict(item) for item in decisions],
    )
    write_csv(output_dir / "joined_evidence.csv", rows)
    write_csv(output_dir / "found_rows.csv", final_rows)
    return {
        "implementation": "trummer_heterogen_v2_3_batched_cascade",
        "mode": "llm",
        "model": f"{args.cheap_model}->{args.expensive_model}",
        "cheap_model": args.cheap_model,
        "expensive_model": args.expensive_model,
        **metrics_dict(metrics),
        "cpu_seconds": cpu_seconds() - started_cpu,
        "engine_seconds": elapsed,
        "wall_seconds": elapsed,
        "llm_calls": metrics.cheap_calls + metrics.expensive_calls,
        "block_join_calls": metrics.expensive_calls,
        "final_answer_rows": len(final_rows),
        "found_movie_ids": sorted(seen),
        "original_movies": len(input_movies),
        "original_reviews": len(input_reviews),
        "pruned_movies": len(movies),
        "pruned_reviews": len(reviews),
        "structured_filters": [item.as_dict() for item in filters],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["suql", "v2_2", "v2_3", "v3"],
    )
    parser.add_argument("--question-dir", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="ollama/gemma4:e2b")
    parser.add_argument("--expensive-model", default="ollama/gemma4:e4b")
    parser.add_argument("--cheap-accept-threshold", type=float, default=3.0)
    parser.add_argument("--cheap-reject-threshold", type=float, default=-1.5)
    parser.add_argument("--cheap-batch-size", type=int, default=8)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--v2-3-expensive-batch-size", type=int, default=32)
    parser.add_argument("--max-expensive-calls", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--token-threshold", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--max-movie-block-size", type=int, default=25)
    parser.add_argument("--max-review-block-size", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = benchmark(args.question_dir)
    if args.method == "suql":
        payload = run_suql(args, spec, output_dir)
    elif args.method == "v2_2":
        payload = run_v2_2(args, spec, output_dir)
    elif args.method == "v2_3":
        payload = run_v2_3(args, spec, output_dir)
    else:
        payload = run_v3(args, spec, output_dir)
    payload["question_dir"] = args.question_dir
    payload["question"] = spec["question"]
    (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
