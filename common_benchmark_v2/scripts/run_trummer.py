#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parent
TRUMMER_ROOT = Path(
    os.environ.get(
        "COMMON_BENCHMARK_TRUMMER_ROOT",
        LAB_ROOT / "project Trummer" / "heterogen_v1",
    )
)


def model_slug(model: str) -> str:
    return model.removeprefix("ollama/").removesuffix(":latest").replace(":", "_").replace("/", "_")


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trummer heterogen_v1 on common benchmark v2.")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="ollama/gemma2:2b")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--token-threshold",
        type=int,
        default=4096,
        help="Combined prompt/output planning budget. V2 uses smaller blocks to avoid hour-long requests.",
    )
    parser.add_argument("--selectivity", type=float, default=0.05)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per block; index-pair output should be much shorter.",
    )
    parser.add_argument("--max-movie-block-size", type=int, default=25)
    parser.add_argument("--max-review-block-size", type=int, default=8)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=3600,
        help="Timeout in seconds for each Trummer LLM block-join request.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir or ROOT / "outputs" / model_slug(args.model) / "trummer_heterogen_v1"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads((ROOT / "benchmark.json").read_text())
    sys.path.insert(0, str(TRUMMER_ROOT))

    from trummer_join.client import ChatClient
    from trummer_join.operators import block_join

    # Unlike v1, Trummer receives every benchmark year. The LLM join predicate
    # must enforce year=1998 together with movie_id=tconst and negative sentiment.
    movies = pd.read_csv(ROOT / "data" / "imdb_structured_joined.csv")
    reviews = pd.read_csv(ROOT / "data" / "imdb_reviews.csv")
    movies = movies.dropna(subset=["movie_id", "title", "year"]).copy()
    reviews = reviews.dropna(subset=["tconst", "review"]).copy()
    movies["text"] = movies.apply(
        lambda row: (
            f"movie_id={row['movie_id']}; title={row['title']}; year={int(row['year'])}; "
            f"director={row.get('director', '')}; runtime={row.get('runtime', '')}; "
            f"genres={row.get('genres', '')}"
        ),
        axis=1,
    )
    reviews["text"] = reviews.apply(
        lambda row: (
            f"tconst={row['tconst']}; review="
            + " ".join(str(row["review"]).replace("<br />", " ").split())[:1400]
        ),
        axis=1,
    )

    if args.dry_run:
        final = movies[
            movies["movie_id"].astype(str).isin(benchmark["ground_truth_movie_ids"])
        ][["movie_id", "title", "year", "runtime", "director", "genres"]].copy()
        stats = pd.DataFrame()
        joined = pd.DataFrame()
        final.to_csv(output_dir / "found_rows.csv", index=False)
        stats.to_csv(output_dir / "join_stats.csv", index=False)
        joined.to_csv(output_dir / "joined_evidence.csv", index=False)
        payload = {
            "implementation": "trummer_heterogen_v1",
            "mode": "dry_run_ground_truth_wiring",
            "model": args.model,
            "api_base": args.api_base,
            "cpu_seconds": 0.0,
            "engine_seconds": 0.0,
            "wall_seconds": 0.0,
            "llm_calls": 0,
            "final_answer_rows": int(len(final)),
            "found_movie_ids": sorted(final["movie_id"].astype(str).unique()),
            "structured_candidates": int(len(movies)),
            "semantic_review_candidates": int(len(reviews)),
            "year_condition_location": "inside_trummer_join_predicate",
            "join_prompt_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    stats, joined = block_join(
        ChatClient(api_base=args.api_base, timeout=args.request_timeout),
        movies,
        reviews,
        benchmark["trummer_join_predicate"],
        args.model,
        selectivity_estimate=args.selectivity,
        token_threshold=args.token_threshold,
        max_completion_tokens=args.max_completion_tokens,
        max_block_1_size=args.max_movie_block_size,
        max_block_2_size=args.max_review_block_size,
        dry_run=False,
    )
    engine_seconds = time.perf_counter() - started_wall
    process_cpu_seconds = cpu_seconds() - started_cpu

    if joined.empty:
        final = pd.DataFrame(columns=["movie_id", "title", "year", "runtime", "director", "genres"])
    else:
        final = joined[
            ["movie_id", "title", "year", "runtime", "director", "genres"]
        ].drop_duplicates().reset_index(drop=True)

    stats.to_csv(output_dir / "join_stats.csv", index=False)
    joined.to_csv(output_dir / "joined_evidence.csv", index=False)
    final.to_csv(output_dir / "found_rows.csv", index=False)
    llm_calls = int(len(stats))
    payload = {
        "implementation": "trummer_heterogen_v1",
        "mode": "llm",
        "model": args.model,
        "api_base": args.api_base,
        "cpu_seconds": process_cpu_seconds,
        "engine_seconds": engine_seconds,
        "wall_seconds": engine_seconds,
        "llm_calls": llm_calls,
        "final_answer_rows": int(len(final)),
        "found_movie_ids": sorted(final["movie_id"].astype(str).unique()),
        "structured_candidates": int(len(movies)),
        "semantic_review_candidates": int(len(reviews)),
        "year_condition_location": "inside_trummer_join_predicate",
        "token_threshold": args.token_threshold,
        "max_completion_tokens": args.max_completion_tokens,
        "max_movie_block_size": args.max_movie_block_size,
        "max_review_block_size": args.max_review_block_size,
        "join_prompt_count": int(len(stats)),
        "prompt_tokens": int(stats["tokens_read"].sum()) if "tokens_read" in stats else 0,
        "completion_tokens": int(stats["tokens_written"].sum()) if "tokens_written" in stats else 0,
    }
    (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
