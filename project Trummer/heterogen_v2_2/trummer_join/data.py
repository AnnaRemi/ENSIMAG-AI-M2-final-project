from __future__ import annotations

from pathlib import Path

import pandas as pd

from .structured_filter import (
    StructuredFilter,
    apply_structured_filters,
    extract_structured_filters,
    prune_movie_frame,
)


HETEROGEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUQL_DATA = HETEROGEN_ROOT.parent.parent / "project SUQL" / "data"


def load_movies_and_reviews(
    data_dir: str | Path | None = None,
    year: int | None = 1998,
    question: str | None = None,
    structured_filters: list[StructuredFilter] | None = None,
    max_movies: int | None = None,
    max_reviews: int | None = None,
    prefilter_reviews_by_movie_id: bool = True,
    api_base: str = "http://127.0.0.1:11434",
    parser_model: str | None = None,
    request_timeout: float = 120.0,
    use_llm_structured_parser: bool = False,
    return_pruning: bool = False,
):
    data_path = Path(data_dir) if data_dir else DEFAULT_SUQL_DATA
    movies_path = data_path / "imdb_structured_joined.csv"
    reviews_path = data_path / "imdb_reviews.csv"

    movies = pd.read_csv(movies_path)
    reviews = pd.read_csv(reviews_path)

    movies = movies.dropna(subset=["movie_id", "title"])
    reviews = reviews.dropna(subset=["tconst", "review"]).copy()

    filters = list(structured_filters or [])
    pruning = None
    if question and use_llm_structured_parser and parser_model:
        movies, pruning = prune_movie_frame(
            movies,
            question,
            api_base=api_base,
            parser_model=parser_model,
            request_timeout=request_timeout,
            use_llm=True,
        )
    elif question:
        filters.extend(extract_structured_filters(question, movies.columns))
        movies = apply_structured_filters(movies, filters)
    elif year is not None:
        filters.append(StructuredFilter("year", "eq", str(year), f"year={year}"))
        movies = apply_structured_filters(movies, filters)
    elif filters:
        movies = apply_structured_filters(movies, filters)

    if max_movies is not None:
        movies = movies.head(max_movies)
    if max_reviews is not None:
        reviews = reviews.head(max_reviews)

    if prefilter_reviews_by_movie_id:
        movie_ids = set(movies["movie_id"].astype(str))
        reviews = reviews[reviews["tconst"].astype(str).isin(movie_ids)].copy()

    movies["join_id"] = movies["movie_id"].astype(str)
    reviews["join_id"] = reviews["tconst"].astype(str)

    movies["text"] = movies.apply(_movie_text, axis=1)
    reviews["text"] = reviews.apply(_review_text, axis=1)

    movies = movies.reset_index(drop=True)
    reviews = reviews.reset_index(drop=True)
    if return_pruning:
        return movies, reviews, pruning
    return movies, reviews


def _movie_text(row: pd.Series) -> str:
    return (
        f"movie_id={row['movie_id']}; "
        f"title={row['title']}; "
        f"year={row['year']}; "
        f"director={row.get('director', '')}; "
        f"runtime={row.get('runtime', '')}; "
        f"genres={row.get('genres', '')}"
    )


def _review_text(row: pd.Series) -> str:
    review = str(row["review"]).replace("<br />", " ")
    review = " ".join(review.split())
    return f"tconst={row['tconst']}; review={review[:1400]}"
