#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from common import LAB_ROOT, ROOT, benchmark, cpu_seconds, load_movies, load_reviews, truth_rows


TRUMMER_ROOT = Path(
    os.environ.get("COMMON_BENCHMARK_V3_HETEROGEN_V1_ROOT", LAB_ROOT / "project Trummer" / "heterogen_v1")
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trummer heterogen_v1 for common benchmark v3.")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="ollama/gemma4:e4b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--token-threshold", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--max-movie-block-size", type=int, default=25)
    parser.add_argument("--max-review-block-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = benchmark()
    movies_list = load_movies()
    reviews_list = load_reviews()
    movies = pd.DataFrame(movies_list)
    reviews = pd.DataFrame(reviews_list)
    sys.path.insert(0, str(TRUMMER_ROOT))
    from trummer_join.client import ChatClient
    from trummer_join.operators import block_join

    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    stats, joined = block_join(
        ChatClient(api_base=args.api_base, timeout=args.request_timeout),
        movies,
        reviews,
        spec["trummer_join_predicate"],
        args.model,
        selectivity_estimate=0.05,
        token_threshold=args.token_threshold,
        max_completion_tokens=args.max_completion_tokens,
        max_block_1_size=args.max_movie_block_size,
        max_block_2_size=args.max_review_block_size,
        dry_run=args.dry_run,
    )
    elapsed = time.perf_counter() - started_wall
    process_cpu = cpu_seconds() - started_cpu

    if args.dry_run:
        final_rows = truth_rows(movies_list)
        joined_rows: list[dict] = []
    else:
        joined_rows = joined.to_dict("records") if not joined.empty else []
        final_rows = []
        seen = set()
        for row in joined_rows:
            movie_id = str(row.get("movie_id", ""))
            if movie_id not in seen:
                seen.add(movie_id)
                final_rows.append(
                    {key: row.get(key, "") for key in ("movie_id", "title", "year", "runtime", "director", "genres")}
                )

    stats.to_csv(output_dir / "join_stats.csv", index=False)
    pd.DataFrame(joined_rows).to_csv(output_dir / "joined_evidence.csv", index=False)
    pd.DataFrame(final_rows).to_csv(output_dir / "found_rows.csv", index=False)
    expensive_seconds = (
        float(stats["seconds"].sum())
        if not args.dry_run and "seconds" in stats
        else 0.0
    )
    payload = {
        "implementation": "trummer_heterogen_v1",
        "mode": "dry_run" if args.dry_run else "llm",
        "model": args.model,
        "cheap_model": "",
        "expensive_model": args.model,
        "cpu_seconds": process_cpu,
        "engine_seconds": elapsed,
        "wall_seconds": elapsed,
        "llm_calls": 0 if args.dry_run else int(len(stats)),
        "block_join_calls": int(len(stats)),
        "cheap_calls": 0,
        "expensive_calls": 0 if args.dry_run else int(len(stats)),
        "planned_cheap_calls": 0,
        "planned_expensive_calls": int(len(stats)),
        "cheap_seconds": 0.0,
        "expensive_seconds": expensive_seconds,
        "cheap_time_percent": 0.0,
        "expensive_time_percent": 100.0 if expensive_seconds > 0 else 0.0,
        "final_answer_rows": len(final_rows),
        "found_movie_ids": sorted({str(row["movie_id"]) for row in final_rows}),
        "input_movies": len(movies_list),
        "input_reviews": len(reviews_list),
        "prompt_tokens": int(stats["tokens_read"].sum()) if "tokens_read" in stats else 0,
        "completion_tokens": int(stats["tokens_written"].sum()) if "tokens_written" in stats else 0,
    }
    (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
