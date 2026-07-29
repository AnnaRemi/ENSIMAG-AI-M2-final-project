#!/usr/bin/env python3
"""Centralize benchmark data and build canonical review/movie/SUQL tables."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
BENCHMARK_ROOT = ROOT / "benchmarks"
BENCHMARK_UNION_ROOT = DATA_ROOT / "benchmark_union"
SUBDATASET_ROOT = DATA_ROOT / "subdatasets"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    question_data_dirs = sorted(BENCHMARK_ROOT.glob("*q/per_question/q_*/data"))
    if not question_data_dirs:
        raise RuntimeError("No benchmark question datasets found")

    review_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    movie_by_id: dict[str, dict[str, str]] = {}

    for source in question_data_dirs:
        suite = source.parents[2].name
        question = source.parent.name
        target = SUBDATASET_ROOT / suite / question
        target.mkdir(parents=True, exist_ok=True)
        for source_file in sorted(source.glob("*.csv")):
            shutil.copy2(source_file, target / source_file.name)

        for row in read_rows(source / "imdb_reviews.csv"):
            movie_id = str(row.get("tconst", "")).strip()
            review = str(row.get("review", "")).strip()
            if movie_id and review:
                review_by_pair.setdefault((movie_id, review), {"tconst": movie_id, "review": review})

        for row in read_rows(source / "imdb_structured_joined.csv"):
            movie_id = str(row.get("movie_id", "")).strip()
            if not movie_id:
                continue
            normalized = {
                "movie_id": movie_id,
                "title": row.get("title", ""),
                "director": row.get("director", ""),
                "year": row.get("year", ""),
                "runtime": row.get("runtime", ""),
                "genres": row.get("genres", ""),
            }
            previous = movie_by_id.get(movie_id)
            if previous is not None and previous != normalized:
                raise ValueError(f"Conflicting structured rows for {movie_id}")
            movie_by_id[movie_id] = normalized

    reviews = sorted(review_by_pair.values(), key=lambda row: (row["tconst"], row["review"]))
    movies = sorted(movie_by_id.values(), key=lambda row: row["movie_id"])
    missing = sorted({row["tconst"] for row in reviews} - movie_by_id.keys())
    if missing:
        raise ValueError(f"Reviews without structured movie rows: {missing[:10]}")

    joined = []
    for review in reviews:
        movie = movie_by_id[review["tconst"]]
        joined.append({**movie, "review": review["review"]})

    write_rows(BENCHMARK_UNION_ROOT / "imdb_reviews.csv", reviews, ["tconst", "review"])
    write_rows(
        BENCHMARK_UNION_ROOT / "imdb_structured.csv",
        movies,
        ["movie_id", "title", "director", "year", "runtime", "genres"],
    )
    # Compatibility name used by Trummer implementations.
    write_rows(
        BENCHMARK_UNION_ROOT / "imdb_structured_joined.csv",
        movies,
        ["movie_id", "title", "director", "year", "runtime", "genres"],
    )
    write_rows(
        BENCHMARK_UNION_ROOT / "imdb_joined.csv",
        joined,
        ["movie_id", "title", "director", "year", "runtime", "genres", "review"],
    )
    print(f"Centralized {len(question_data_dirs)} question datasets")
    print(f"Benchmark-union structured movies: {len(movies)}")
    print(f"Benchmark-union unique reviews: {len(reviews)}")
    print(f"Benchmark-union SUQL joined rows: {len(joined)}")


if __name__ == "__main__":
    main()
