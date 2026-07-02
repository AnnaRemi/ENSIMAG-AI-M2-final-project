#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from trummer_join.cascade import (
    Candidate,
    CascadeConfig,
    CascadeJoin,
    metrics_dict,
)


DEFAULT_PREDICATE = (
    "The movie and review refer to the same title based on movie_id/tconst, "
    "and the review expresses an overall negative, critical, or strongly "
    "unfavorable opinion of the movie"
)
LOCAL_DATA = Path(__file__).resolve().parents[2] / "common_benchmark_v2" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run heterogen_v2_3 with batched cheap routing."
    )
    parser.add_argument("--data-dir", default=str(LOCAL_DATA))
    parser.add_argument("--year", type=int, default=1998)
    parser.add_argument("--predicate", default=DEFAULT_PREDICATE)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="gemma4:e2b")
    parser.add_argument("--expensive-model", default="gemma4:e4b")
    parser.add_argument("--cascade-target", type=float, default=0.9)
    parser.add_argument("--calibration-budget", type=int, default=20)
    parser.add_argument("--cheap-batch-size", type=int, default=8)
    parser.add_argument("--expensive-batch-size", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--max-movies", type=int)
    parser.add_argument("--max-reviews", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs/local")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    movies = load_movies(
        Path(args.data_dir) / "imdb_structured_joined.csv",
        args.year,
        args.max_movies,
    )
    reviews = load_reviews(
        Path(args.data_dir) / "imdb_reviews.csv",
        args.max_reviews,
    )
    config = CascadeConfig(
        api_base=args.api_base,
        cheap_model=args.cheap_model,
        expensive_model=args.expensive_model,
        cascade_target=args.cascade_target,
        calibration_budget=args.calibration_budget,
        cheap_batch_size=args.cheap_batch_size,
        expensive_batch_size=args.expensive_batch_size,
        request_timeout=args.request_timeout,
    )
    join = (
        CascadeJoin(
            config,
            cheap_score_batch=dry_run_score_batch,
            expensive_classify=dry_run_expensive,
        )
        if args.dry_run
        else CascadeJoin(config)
    )
    rows, decisions, metrics = join.run(movies, reviews, args.predicate)
    write_csv(output_dir / "joined_evidence.csv", rows)
    write_csv(
        output_dir / "cascade_decisions.csv",
        [asdict(item) for item in decisions],
    )
    final = deduplicate_movies(rows)
    write_csv(output_dir / "final_movies.csv", final)
    payload = {
        "implementation": "trummer_heterogen_v2_3_batched_cascade",
        "mode": "dry_run" if args.dry_run else "llm",
        "predicate": args.predicate,
        "config": asdict(config),
        **metrics_dict(metrics),
        "final_answer_rows": len(final),
        "found_movie_ids": sorted(row["movie_id"] for row in final),
    }
    (output_dir / "run_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


def load_movies(path: Path, year: int, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                if int(float(row.get("year", ""))) != year:
                    continue
            except ValueError:
                continue
            row["text"] = (
                f"movie_id={row.get('movie_id', '')}; title={row.get('title', '')}; "
                f"year={row.get('year', '')}; director={row.get('director', '')}; "
                f"runtime={row.get('runtime', '')}; genres={row.get('genres', '')}"
            )
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_reviews(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            review = " ".join(
                (row.get("review") or "").replace("<br />", " ").split()
            )
            row["review"] = review
            row["text"] = f"tconst={row.get('tconst', '')}; review={review}"
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def dry_run_score_batch(
    candidates: list[Candidate],
    predicate: str,
) -> dict[int, float]:
    del predicate
    negative = (
        "bad", "poor", "awful", "boring", "disappoint", "worst", "hate",
        "waste", "stupid", "terrible", "dull", "unwatchable",
    )
    return {
        candidate.candidate_id: (
            2.0
            if any(
                term in candidate.review.get("review", "").lower()
                for term in negative
            )
            else -2.0
        )
        for candidate in candidates
    }


def dry_run_expensive(
    candidates: list[Candidate],
    predicate: str,
) -> set[int]:
    return {
        candidate_id
        for candidate_id, score in dry_run_score_batch(
            candidates,
            predicate,
        ).items()
        if score > 0
    }


def deduplicate_movies(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for row in rows:
        if row["movie_id"] not in seen:
            seen.add(row["movie_id"])
            result.append(row)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
