#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trummer_join.client import ChatClient
from trummer_join.data import DEFAULT_SUQL_DATA, load_movies_and_reviews
from trummer_join.operators import adaptive_join, block_join


DEFAULT_PREDICATE = (
    "the review chunk is about the same movie row, based on movie_id/tconst, "
    "and the review expresses a negative, critical, or strongly unfavorable opinion about the movie"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Trummer-style semantic block join over local IMDb movies and reviews."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_SUQL_DATA), help="Directory containing IMDb CSV tables.")
    parser.add_argument("--year", type=int, default=1998, help="Structured movie year filter.")
    parser.add_argument("--predicate", default=DEFAULT_PREDICATE, help="Natural-language join predicate.")
    parser.add_argument("--api-base", default="http://localhost:11434", help="Ollama or OpenAI-compatible API base.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default="ollama/phi4-mini", help="Chat model name.")
    parser.add_argument("--operator", choices=["block", "adaptive"], default="adaptive")
    parser.add_argument("--selectivity", type=float, default=0.001, help="Initial selectivity estimate.")
    parser.add_argument("--token-threshold", type=int, default=4000, help="Per-request token budget.")
    parser.add_argument("--token-model", default="gpt-4o", help="Tokenizer model for prompt sizing.")
    parser.add_argument("--max-movies", type=int, default=None, help="Limit movie rows after year filtering.")
    parser.add_argument("--max-reviews", type=int, default=None, help="Limit review rows for experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls and run deterministic wiring check.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV outputs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    movies, reviews = load_movies_and_reviews(
        data_dir=args.data_dir,
        year=args.year,
        max_movies=args.max_movies,
        max_reviews=args.max_reviews,
    )
    print(f"Loaded {len(movies)} movie rows for year={args.year}")
    print(f"Loaded {len(reviews)} review chunks")
    print(f"Predicate: {args.predicate}")

    movies.to_csv(output_dir / "use_case3_movie_candidates.csv", index=False)

    client = ChatClient(api_base=args.api_base, api_key=args.api_key)
    if args.operator == "block":
        stats, joined = block_join(
            client,
            movies,
            reviews,
            args.predicate,
            args.model,
            selectivity_estimate=args.selectivity,
            token_threshold=args.token_threshold,
            token_model=args.token_model,
            dry_run=args.dry_run,
        )
    else:
        stats, joined = adaptive_join(
            client,
            movies,
            reviews,
            args.predicate,
            args.model,
            initial_selectivity=args.selectivity,
            token_threshold=args.token_threshold,
            token_model=args.token_model,
            dry_run=args.dry_run,
        )

    if not joined.empty:
        joined = _deduplicate_joined(joined)
        final = joined[["movie_id", "title", "year", "director", "genres", "review"]].copy()
        final["review"] = final["review"].astype(str).str.replace("<br />", " ", regex=False).str.slice(0, 500)
    else:
        final = pd.DataFrame(columns=["movie_id", "title", "year", "director", "genres", "review"])

    stats.to_csv(output_dir / "use_case3_join_stats.csv", index=False)
    joined.to_csv(output_dir / "use_case3_joined_pairs.csv", index=False)
    final.to_csv(output_dir / "use_case3_final_movies.csv", index=False)

    print(f"Wrote stats: {output_dir / 'use_case3_join_stats.csv'}")
    print(f"Wrote joined evidence: {output_dir / 'use_case3_joined_pairs.csv'}")
    print(f"Wrote final movie table: {output_dir / 'use_case3_final_movies.csv'}")
    print(f"Matched {len(joined)} movie-review pairs and {final['movie_id'].nunique() if not final.empty else 0} movies")


def _deduplicate_joined(joined: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["movie_id", "tconst", "review"] if col in joined.columns]
    if cols:
        return joined.drop_duplicates(subset=cols).reset_index(drop=True)
    return joined.drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    main()
