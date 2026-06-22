from __future__ import annotations

from pathlib import Path

import pandas as pd


HETEROGEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUQL_DATA = HETEROGEN_ROOT.parent.parent / "project SUQL" / "data"


def load_movies_and_reviews(
    data_dir: str | Path | None = None,
    year: int = 1998,
    max_movies: int | None = None,
    max_reviews: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir) if data_dir else DEFAULT_SUQL_DATA
    movies_path = data_path / "imdb_structured_joined.csv"
    reviews_path = data_path / "imdb_reviews.csv"

    movies = pd.read_csv(movies_path)
    reviews = pd.read_csv(reviews_path)

    movies = movies[movies["year"] == year].copy()
    movies = movies.dropna(subset=["movie_id", "title"])
    reviews = reviews.dropna(subset=["tconst", "review"]).copy()

    if max_movies is not None:
        movies = movies.head(max_movies)
    if max_reviews is not None:
        reviews = reviews.head(max_reviews)

    movies["join_id"] = movies["movie_id"].astype(str)
    reviews["join_id"] = reviews["tconst"].astype(str)

    movies["text"] = movies.apply(_movie_text, axis=1)
    reviews["text"] = reviews.apply(_review_text, axis=1)

    return movies.reset_index(drop=True), reviews.reset_index(drop=True)


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
