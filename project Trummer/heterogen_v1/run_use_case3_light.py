#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from pathlib import Path


DEFAULT_PREDICATE = (
    "the review chunk is about the same movie row, based on movie_id/tconst, "
    "and the review expresses a negative, critical, or strongly unfavorable opinion about the movie"
)
PAIR_RE = re.compile(r"(\d+)\s*,\s*(\d+)")
LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "project SUQL" / "data"
DEFAULT_DATA_DIR = Path("data") if Path("data").is_dir() else LOCAL_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dependency-light Trummer-style block join for Aker nodes without pandas/numpy."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--year", type=int, default=1998)
    parser.add_argument("--predicate", default=DEFAULT_PREDICATE)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="gemma2:2b")
    parser.add_argument("--movie-block-size", type=int, default=4)
    parser.add_argument("--review-block-size", type=int, default=8)
    parser.add_argument("--max-movies", type=int, default=None)
    parser.add_argument("--max-reviews", type=int, default=None)
    parser.add_argument(
        "--prefilter-reviews-by-movie-id",
        action="store_true",
        help="After sampling reviews, keep only rows whose tconst matches a selected movie_id.",
    )
    parser.add_argument(
        "--sample-movies-before-year-filter",
        action="store_true",
        help="Take the first --max-movies rows overall before applying --year.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    movies = load_movies(
        data_dir / "imdb_structured_joined.csv",
        args.year,
        args.max_movies,
        sample_before_year_filter=args.sample_movies_before_year_filter,
    )
    reviews = load_reviews(data_dir / "imdb_reviews.csv", args.max_reviews)
    sampled_reviews = len(reviews)
    if args.prefilter_reviews_by_movie_id:
        movie_ids = {movie.get("movie_id", "") for movie in movies}
        reviews = [review for review in reviews if review.get("tconst", "") in movie_ids]
    print(f"Loaded {len(movies)} movie rows for year={args.year}")
    print(f"Sampled {sampled_reviews} review chunks overall")
    print(f"Semantic-join candidate reviews after movie_id prefilter: {len(reviews)}")

    stats, joined = run_block_join(
        movies,
        reviews,
        predicate=args.predicate,
        api_base=args.api_base,
        model=args.model,
        movie_block_size=args.movie_block_size,
        review_block_size=args.review_block_size,
        dry_run=args.dry_run,
    )

    joined = deduplicate(joined)
    if not joined:
        joined = deduplicate(dry_run_pairs(movies, reviews, source="deterministic_fallback"))
        if joined:
            print(f"LLM returned no pairs; added {len(joined)} deterministic fallback pairs.")
    write_csv(out_dir / "use_case3_join_stats.csv", stats)
    write_csv(out_dir / "use_case3_joined_pairs.csv", joined)
    write_csv(out_dir / "use_case3_final_movies.csv", final_rows(joined))
    print(f"Wrote {out_dir / 'use_case3_join_stats.csv'}")
    print(f"Wrote {out_dir / 'use_case3_joined_pairs.csv'}")
    print(f"Wrote {out_dir / 'use_case3_final_movies.csv'}")
    print(f"Matched {len(joined)} movie-review pairs")


def run_block_join(
    movies: list[dict[str, str]],
    reviews: list[dict[str, str]],
    *,
    predicate: str,
    api_base: str,
    model: str,
    movie_block_size: int,
    review_block_size: int,
    dry_run: bool,
    match_source: str = "llm",
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    stats: list[dict[str, object]] = []
    joined: list[dict[str, str]] = []
    movie_blocks = list(blocks(movies, movie_block_size))
    review_blocks = list(blocks(reviews, review_block_size))
    for movie_idx, movie_block in enumerate(movie_blocks, 1):
        for review_idx, review_block in enumerate(review_blocks, 1):
            print(f"Joining movie block {movie_idx}/{len(movie_blocks)} with review block {review_idx}/{len(review_blocks)}")
            started = time.time()
            prompt = create_prompt(movie_block, review_block, predicate)
            if dry_run:
                matches = dry_run_pairs(movie_block, review_block, source=match_source)
                prompt_tokens = approx_tokens(prompt)
                completion_tokens = 0
            else:
                answer, prompt_tokens, completion_tokens = ollama_chat(api_base, model, prompt)
                matches = parse_pairs(answer, movie_block, review_block, source=match_source)
            joined.extend(matches)
            stats.append(
                {
                    "stage": "block_join",
                    "movie_block": movie_idx,
                    "review_block": review_idx,
                    "seconds": f"{time.time() - started:.3f}",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "matches": len(matches),
                }
            )
    return stats, joined


def load_movies(
    path: Path,
    year: int,
    limit: int | None,
    sample_before_year_filter: bool = False,
) -> list[dict[str, str]]:
    rows = []
    source_rows_seen = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_rows_seen += 1
            if sample_before_year_filter and limit is not None and source_rows_seen > limit:
                break
            if int(row.get("year") or 0) != year:
                continue
            row["text"] = (
                f"movie_id={row.get('movie_id', '')}; "
                f"title={row.get('title', '')}; "
                f"year={row.get('year', '')}; "
                f"director={row.get('director', '')}; "
                f"runtime={row.get('runtime', '')}; "
                f"genres={row.get('genres', '')}"
            )
            rows.append(row)
            if not sample_before_year_filter and limit is not None and len(rows) >= limit:
                break
    return rows


def load_reviews(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            review = " ".join((row.get("review") or "").replace("<br />", " ").split())
            row["review"] = review
            row["text"] = f"tconst={row.get('tconst', '')}; review={review[:1400]}"
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def create_prompt(block_1: list[dict[str, str]], block_2: list[dict[str, str]], predicate: str) -> str:
    parts = [
        "Find indexes x,y where x is the number of an entry in Collection 1 "
        "and y is the number of an entry in Collection 2 such that the pair "
        f"satisfies this predicate: {predicate}",
        "Collection 1 contains structured movie rows. Collection 2 contains review chunks.",
        "Return every matching pair as x,y separated by semicolons.",
        'Write "Finished" after the last pair.',
        "Do not explain.",
        "",
        "Collection 1:",
    ]
    for idx, row in enumerate(block_1, 1):
        parts.append(f"{idx}: {row['text']}")
    parts += ["", "Collection 2:"]
    for idx, row in enumerate(block_2, 1):
        parts.append(f"{idx}: {row['text']}")
    parts += ["", "Index pairs:"]
    return "\n".join(parts)


def ollama_chat(api_base: str, model: str, prompt: str) -> tuple[str, int, int]:
    payload = {
        "model": model.removeprefix("ollama/"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200, "stop": ["Finished"]},
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))
    return (
        data.get("message", {}).get("content", ""),
        int(data.get("prompt_eval_count") or approx_tokens(prompt)),
        int(data.get("eval_count") or 0),
    )


def parse_pairs(
    answer: str,
    movies: list[dict[str, str]],
    reviews: list[dict[str, str]],
    source: str = "llm",
) -> list[dict[str, str]]:
    rows = []
    for raw_x, raw_y in PAIR_RE.findall(answer):
        x = int(raw_x) - 1
        y = int(raw_y) - 1
        if 0 <= x < len(movies) and 0 <= y < len(reviews):
            rows.append(joined_row(movies[x], reviews[y], source=source))
    return rows


def dry_run_pairs(
    movies: list[dict[str, str]],
    reviews: list[dict[str, str]],
    source: str = "dry_run",
) -> list[dict[str, str]]:
    terms = ("bad", "poor", "stupid", "trash", "awful", "boring", "disappoint", "unbelievable", "worst", "hate")
    rows = []
    for movie in movies:
        for review in reviews:
            if movie.get("movie_id") == review.get("tconst") and any(term in review.get("review", "").lower() for term in terms):
                rows.append(joined_row(movie, review, source=source))
    return rows


def joined_row(
    movie: dict[str, str],
    review: dict[str, str],
    source: str = "llm",
) -> dict[str, str]:
    return {
        "movie_id": movie.get("movie_id", ""),
        "title": movie.get("title", ""),
        "year": movie.get("year", ""),
        "director": movie.get("director", ""),
        "runtime": movie.get("runtime", ""),
        "genres": movie.get("genres", ""),
        "tconst": review.get("tconst", ""),
        "review": review.get("review", ""),
        "match_source": source,
    }


def blocks(rows: list[dict[str, str]], block_size: int):
    for start in range(0, len(rows), block_size):
        yield rows[start : start + block_size]


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    unique = []
    for row in rows:
        key = (row.get("movie_id"), row.get("tconst"), row.get("review"))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def final_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "movie_id": row.get("movie_id", ""),
            "title": row.get("title", ""),
            "year": row.get("year", ""),
            "director": row.get("director", ""),
            "genres": row.get("genres", ""),
            "review": row.get("review", "")[:500],
            "match_source": row.get("match_source", ""),
        }
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        if "stats" in path.name:
            fieldnames = ["movie_block", "review_block", "seconds", "prompt_tokens", "completion_tokens", "matches"]
        elif "final" in path.name:
            fieldnames = ["movie_id", "title", "year", "director", "genres", "review", "match_source"]
        else:
            fieldnames = ["movie_id", "title", "year", "director", "runtime", "genres", "tconst", "review", "match_source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


if __name__ == "__main__":
    main()
