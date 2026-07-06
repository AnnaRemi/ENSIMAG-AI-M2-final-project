#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from trummer_join.cascade import CascadeConfig, CascadeJoin, Candidate, metrics_dict
from trummer_join.structured_filter import (
    StructuredFilter,
    StructuredPruningResult,
    prune_movie_frame,
    semantic_predicate_from_question,
)


DEFAULT_QUESTION = (
    "Which movies released in 1998 have reviews expressing an overall negative, "
    "critical, or strongly unfavorable opinion of the movie?"
)
LOCAL_DATA = Path(__file__).resolve().parents[3] / "common_benchmark_v2" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the heterogen_v3 structured-pruned cascading Trummer join.")
    parser.add_argument("--data-dir", default=str(LOCAL_DATA))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--year", type=int, default=1998, help="Fallback year filter when --question is empty.")
    parser.add_argument("--predicate", default=None, help="Override the semantic predicate derived from --question.")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="gemma4:e2b")
    parser.add_argument("--expensive-model", default="gemma4:e4b")
    parser.add_argument("--structured-parser-model")
    parser.add_argument("--disable-llm-structured-parser", action="store_true")
    parser.add_argument("--cascade-target", type=float, default=0.9)
    parser.add_argument("--calibration-budget", type=int, default=20)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--max-expensive-calls", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--max-movies", type=int)
    parser.add_argument("--max-reviews", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs/local")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_movies = load_movies(data_dir / "imdb_structured_joined.csv", args.max_movies)
    input_reviews = load_reviews(data_dir / "imdb_reviews.csv", args.max_reviews)
    movies, reviews, structured_filters = prune_inputs(
        input_movies,
        input_reviews,
        args.question,
        args.year,
        api_base=args.api_base,
        parser_model=args.structured_parser_model or args.cheap_model,
        request_timeout=args.request_timeout,
        use_llm=not args.dry_run and not args.disable_llm_structured_parser,
    )
    predicate = args.predicate or structured_filters.semantic_predicate or semantic_predicate_from_question(args.question)
    print(
        f"Loaded movies={len(input_movies)}, reviews={len(input_reviews)}; "
        f"after structured pruning movies={len(movies)}, reviews={len(reviews)}",
        flush=True,
    )

    config = CascadeConfig(
        api_base=args.api_base,
        cheap_model=args.cheap_model,
        expensive_model=args.expensive_model,
        cascade_target=args.cascade_target,
        calibration_budget=args.calibration_budget,
        expensive_batch_size=args.expensive_batch_size,
        max_expensive_calls=args.max_expensive_calls,
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

    rows, decisions, metrics = join.run(movies, reviews, predicate)
    write_csv(output_dir / "joined_evidence.csv", rows)
    write_csv(output_dir / "cascade_decisions.csv", [asdict(item) for item in decisions])
    final = deduplicate_movies(rows)
    write_csv(output_dir / "final_movies.csv", final)
    payload = {
        "implementation": "trummer_heterogen_v3_pruned_cascade",
        "mode": "dry_run" if args.dry_run else "llm",
        "predicate": predicate,
        "question": args.question,
        "structured_filters": [item.as_dict() for item in structured_filters.filters],
        "structured_pruning": structured_filters.as_dict(),
        "config": asdict(config),
        **metrics_dict(metrics),
        "original_movies": len(input_movies),
        "original_reviews": len(input_reviews),
        "pruned_movies": len(movies),
        "pruned_reviews": len(reviews),
        "final_answer_rows": len(final),
        "found_movie_ids": sorted({row["movie_id"] for row in final}),
    }
    (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Outputs: {output_dir.resolve()}", flush=True)


def load_movies(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["text"] = (
                f"movie_id={row.get('movie_id', '')}; title={row.get('title', '')}; "
                f"year={row.get('year', '')}; director={row.get('director', '')}; "
                f"runtime={row.get('runtime', '')}; genres={row.get('genres', '')}"
            )
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def prune_inputs(
    movies: list[dict[str, str]],
    reviews: list[dict[str, str]],
    question: str,
    fallback_year: int | None,
    *,
    api_base: str = "http://127.0.0.1:11434",
    parser_model: str | None = None,
    request_timeout: float = 120.0,
    use_llm: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, str]], StructuredPruningResult]:
    movie_frame = pd.DataFrame(movies)
    review_frame = pd.DataFrame(reviews)
    pruned_movies, pruning = prune_movie_frame(
        movie_frame,
        question,
        api_base=api_base,
        parser_model=parser_model,
        request_timeout=request_timeout,
        use_llm=use_llm,
    )
    if not pruning.filters and fallback_year is not None and "year" in movie_frame.columns:
        fallback_filter = StructuredFilter("year", "eq", str(fallback_year), str(fallback_year))
        pruned_movies = movie_frame[
            pd.to_numeric(movie_frame["year"], errors="coerce") == fallback_year
        ].reset_index(drop=True)
        pruning = StructuredPruningResult(
            mode="fallback_year",
            filters=[fallback_filter],
            semantic_predicate=semantic_predicate_from_question(question),
        )
    movie_ids = set(pruned_movies["movie_id"].astype(str)) if "movie_id" in pruned_movies else set()
    pruned_reviews = pd.DataFrame(
        [review for review in reviews if str(review.get("tconst", "")) in movie_ids],
        columns=review_frame.columns,
    ).reset_index(drop=True)
    return (
        pruned_movies.to_dict("records"),
        pruned_reviews.to_dict("records"),
        pruning,
    )


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
