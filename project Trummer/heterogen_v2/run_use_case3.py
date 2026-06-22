#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from trummer_join.cascade import CascadeConfig, CascadeJoin, Candidate, metrics_dict


DEFAULT_PREDICATE = (
    "the review is about the same movie based on movie_id/tconst and expresses "
    "an overall negative, critical, or strongly unfavorable opinion of the movie"
)
LOCAL_DATA = Path(__file__).resolve().parents[2] / "common_benchmark_v2" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the heterogen_v2 cascading Trummer join.")
    parser.add_argument("--data-dir", default=str(LOCAL_DATA))
    parser.add_argument("--year", type=int, default=1998)
    parser.add_argument("--predicate", default=DEFAULT_PREDICATE)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="gemma2:2b")
    parser.add_argument("--expensive-model", default="qwen2.5:3b")
    parser.add_argument("--cheap-accept-threshold", type=float, default=3.0)
    parser.add_argument("--cheap-reject-threshold", type=float, default=-1.5)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--max-movies", type=int)
    parser.add_argument("--max-reviews", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs/local")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    movies = load_movies(data_dir / "imdb_structured_joined.csv", args.year, args.max_movies)
    reviews = load_reviews(data_dir / "imdb_reviews.csv", args.max_reviews)
    print(f"Loaded movies={len(movies)}, reviews={len(reviews)}, year={args.year}", flush=True)

    config = CascadeConfig(
        api_base=args.api_base,
        cheap_model=args.cheap_model,
        expensive_model=args.expensive_model,
        accept_threshold=args.cheap_accept_threshold,
        reject_threshold=args.cheap_reject_threshold,
        expensive_batch_size=args.expensive_batch_size,
        request_timeout=args.request_timeout,
    )
    if args.dry_run:
        join = CascadeJoin(
            config,
            cheap_score=dry_run_score,
            expensive_classify=dry_run_expensive,
        )
    else:
        join = CascadeJoin(config)

    rows, decisions, metrics = join.run(movies, reviews, args.predicate)
    write_csv(output_dir / "joined_evidence.csv", rows)
    write_csv(output_dir / "cascade_decisions.csv", [asdict(item) for item in decisions])
    final = deduplicate_movies(rows)
    write_csv(output_dir / "final_movies.csv", final)
    payload = {
        "implementation": "trummer_heterogen_v2_cascade",
        "mode": "dry_run" if args.dry_run else "llm",
        "predicate": args.predicate,
        "config": asdict(config),
        **metrics_dict(metrics),
        "final_answer_rows": len(final),
        "found_movie_ids": sorted({row["movie_id"] for row in final}),
    }
    (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Outputs: {output_dir.resolve()}", flush=True)


def load_movies(path: Path, year: int, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(float(row.get("year") or 0)) != year:
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
            review = " ".join((row.get("review") or "").replace("<br />", " ").split())
            row["review"] = review
            row["text"] = f"tconst={row.get('tconst', '')}; review={review}"
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def dry_run_score(candidate: Candidate, predicate: str) -> float:
    del predicate
    text = candidate.review.get("review", "").lower()
    negative = (
        "bad", "poor", "awful", "boring", "disappoint", "worst", "hate",
        "waste", "stupid", "terrible", "dull", "unwatchable",
    )
    return 2.0 if any(term in text for term in negative) else -2.0


def dry_run_expensive(candidates: list[Candidate], predicate: str) -> set[int]:
    return {
        candidate.candidate_id
        for candidate in candidates
        if dry_run_score(candidate, predicate) > 0
    }


def deduplicate_movies(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    seen = set()
    for row in rows:
        if row["movie_id"] in seen:
            continue
        seen.add(row["movie_id"])
        result.append(
            {
                key: row.get(key, "")
                for key in ("movie_id", "title", "year", "director", "runtime", "genres", "match_source")
            }
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["movie_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
