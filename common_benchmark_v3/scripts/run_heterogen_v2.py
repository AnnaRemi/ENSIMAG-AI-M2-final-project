#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from common import LAB_ROOT, benchmark, cpu_seconds, load_movies, load_reviews, truth_rows, write_csv


TRUMMER_ROOT = Path(
    os.environ.get("COMMON_BENCHMARK_V3_HETEROGEN_V2_ROOT", LAB_ROOT / "project Trummer" / "heterogen_v2")
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trummer heterogen_v2 cascade for common benchmark v3.")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="ollama/gemma4:e2b")
    parser.add_argument("--expensive-model", default="ollama/gemma4:e4b")
    parser.add_argument("--cascade-target", type=float, default=0.9)
    parser.add_argument("--calibration-budget", type=int, default=20)
    parser.add_argument("--manual-confidence-threshold", type=float)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(TRUMMER_ROOT))
    from trummer_join.cascade import CascadeConfig, CascadeJoin, Candidate, metrics_dict

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = benchmark()
    movies = load_movies()
    reviews = load_reviews()
    truth = set(spec["ground_truth_movie_ids"])
    config = CascadeConfig(
        api_base=args.api_base,
        cheap_model=args.cheap_model,
        expensive_model=args.expensive_model,
        cascade_target=args.cascade_target,
        calibration_budget=args.calibration_budget,
        manual_confidence_threshold=args.manual_confidence_threshold,
        expensive_batch_size=args.expensive_batch_size,
        request_timeout=args.request_timeout,
    )

    if args.dry_run:
        join = CascadeJoin(
            config,
            cheap_score=lambda candidate, predicate: 0.0,
            expensive_classify=lambda candidates, predicate: {
                candidate.candidate_id
                for candidate in candidates
                if candidate.movie.get("movie_id") in truth
            },
        )
    else:
        join = CascadeJoin(config)

    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    rows, decisions, metrics = join.run(movies, reviews, spec["trummer_join_predicate"])
    elapsed = time.perf_counter() - started_wall
    process_cpu = cpu_seconds() - started_cpu
    if args.dry_run:
        final_rows = truth_rows(movies)
        evidence_rows: list[dict] = []
    else:
        evidence_rows = rows
        final_rows = []
        seen = set()
        for row in rows:
            movie_id = row.get("movie_id", "")
            if movie_id not in seen:
                seen.add(movie_id)
                final_rows.append(
                    {key: row.get(key, "") for key in ("movie_id", "title", "year", "runtime", "director", "genres", "match_source")}
                )

    write_csv(output_dir / "cascade_decisions.csv", [asdict(item) for item in decisions])
    write_csv(output_dir / "joined_evidence.csv", evidence_rows)
    write_csv(output_dir / "found_rows.csv", final_rows)
    payload = {
        "implementation": "trummer_heterogen_v2_cascade",
        "mode": "dry_run" if args.dry_run else "llm",
        "model": f"{args.cheap_model}->{args.expensive_model}",
        "cheap_model": args.cheap_model,
        "expensive_model": args.expensive_model,
        **metrics_dict(metrics),
        "cpu_seconds": process_cpu,
        "engine_seconds": elapsed,
        "wall_seconds": elapsed,
        "llm_calls": 0 if args.dry_run else metrics.cheap_calls + metrics.expensive_calls,
        "block_join_calls": 0,
        "cheap_calls": 0 if args.dry_run else metrics.cheap_calls,
        "expensive_calls": 0 if args.dry_run else metrics.expensive_calls,
        "planned_cheap_calls": metrics.cheap_calls,
        "planned_expensive_calls": metrics.expensive_calls,
        "final_answer_rows": len(final_rows),
        "found_movie_ids": sorted({str(row["movie_id"]) for row in final_rows}),
    }
    (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
