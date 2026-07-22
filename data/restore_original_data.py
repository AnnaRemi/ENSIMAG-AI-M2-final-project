#!/usr/bin/env python3
"""Restore the original project datasets from Stanford IMDb-ID and IMDb TSVs.

Reimplements the notebooks committed at 63438c90362fe5fa9b93f0292beb716987d8aec5:
  project SUQL/data/data_preprocessing.ipynb
  project SUQL/data/structured_imdb_join.ipynb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "sources"
OUTPUT_ROOT = DATA_ROOT / "canonical"
NOTEBOOK_COMMIT = "63438c90362fe5fa9b93f0292beb716987d8aec5"
OUTPUT_COLUMNS = ["movie_id", "title", "director", "year", "runtime", "genres"]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    return path


def resolve_directors(value: object, name_map: dict[str, str]) -> object:
    if pd.isna(value):
        return pd.NA
    names = [name_map.get(identifier.strip()) for identifier in str(value).split(",")]
    names = [name for name in names if name]
    return ", ".join(dict.fromkeys(names)) if names else pd.NA


def main() -> None:
    options = args()
    raw = options.raw_dir.resolve()
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    reviews_path = require(raw / "imdb-id-train.parquet")
    basics_path = require(raw / "title.basics.tsv.gz")
    crew_path = require(raw / "title.crew.tsv.gz")
    names_path = require(raw / "name.basics.tsv.gz")

    reviews = pd.read_parquet(reviews_path)
    reviews = reviews.rename(columns={"movie_id": "tconst", "text": "review"})
    required_review_columns = {"tconst", "review"}
    if not required_review_columns <= set(reviews.columns):
        raise ValueError(f"Review columns missing: {required_review_columns - set(reviews.columns)}")
    reviews["tconst"] = reviews["tconst"].astype(str).str.strip()
    # Preserve all 25,000 source rows exactly as the recovered notebook did.
    # Duplicate pairs are measured during validation, not removed from source data.
    reviews = reviews.dropna(subset=["tconst", "review"])
    review_ids = set(reviews["tconst"])

    movie_parts: list[pd.DataFrame] = []
    review_basic_parts: list[pd.DataFrame] = []
    basics_columns = ["tconst", "titleType", "primaryTitle", "startYear", "runtimeMinutes", "genres"]
    for chunk in pd.read_csv(
        basics_path, sep="\t", usecols=basics_columns, dtype=str,
        na_values="\\N", keep_default_na=False, chunksize=500_000,
    ):
        review_rows = chunk.loc[chunk["tconst"].isin(review_ids), basics_columns[0:1] + basics_columns[2:]]
        if not review_rows.empty:
            review_basic_parts.append(review_rows)
        movies = chunk.loc[chunk["titleType"].eq("movie"), basics_columns[0:1] + basics_columns[2:]].copy()
        movies = movies.rename(columns={
            "tconst": "movie_id", "primaryTitle": "title",
            "startYear": "year", "runtimeMinutes": "runtime",
        })
        movies = movies.dropna(subset=["movie_id", "title", "year", "runtime", "genres"])
        movies = movies.loc[movies[["movie_id", "title", "year", "runtime", "genres"]].ne("").all(axis=1)]
        if not movies.empty:
            movie_parts.append(movies[["movie_id", "title", "year", "runtime", "genres"]])
    movie_basics = pd.concat(movie_parts, ignore_index=True)
    review_basics = pd.concat(review_basic_parts, ignore_index=True)
    movie_ids = set(movie_basics["movie_id"])

    crew_parts: list[pd.DataFrame] = []
    relevant_ids = movie_ids | review_ids
    for chunk in pd.read_csv(
        crew_path, sep="\t", usecols=["tconst", "directors"], dtype=str,
        na_values="\\N", keep_default_na=False, chunksize=1_000_000,
    ):
        matched = chunk.loc[chunk["tconst"].isin(relevant_ids), ["tconst", "directors"]]
        if not matched.empty:
            crew_parts.append(matched)
    crew = pd.concat(crew_parts, ignore_index=True)
    director_ids = set(
        crew["directors"].dropna().str.split(",").explode().astype(str).str.strip()
    ) - {""}

    name_map: dict[str, str] = {}
    for chunk in pd.read_csv(
        names_path, sep="\t", usecols=["nconst", "primaryName"], dtype=str,
        na_values="\\N", keep_default_na=False, chunksize=500_000,
    ):
        matched = chunk.loc[chunk["nconst"].isin(director_ids)]
        name_map.update(zip(matched["nconst"], matched["primaryName"]))
    crew["director"] = crew["directors"].map(lambda value: resolve_directors(value, name_map))

    structured_crew = crew.rename(columns={"tconst": "movie_id"})
    structured_crew = structured_crew.dropna(subset=["director"])[["movie_id", "director"]]
    structured = movie_basics.merge(structured_crew, on="movie_id", how="inner")
    structured = structured[OUTPUT_COLUMNS].sort_values("movie_id").drop_duplicates("movie_id")
    structured[["year", "runtime"]] = structured[["year", "runtime"]].astype(int)

    review_crew = crew[["tconst", "director"]]
    joined = reviews.merge(review_basics, on="tconst", how="left").merge(review_crew, on="tconst", how="left")
    joined = joined.rename(columns={
        "tconst": "movie_id", "primaryTitle": "title",
        "startYear": "year", "runtimeMinutes": "runtime",
    })
    joined = joined[[*OUTPUT_COLUMNS, "review"]]

    review_columns = [column for column in ("tconst", "review", "score", "label") if column in reviews.columns]
    temporary = output / ".restore_tmp"
    temporary.mkdir(exist_ok=True)
    reviews[review_columns].to_csv(temporary / "imdb_reviews.csv", index=False)
    structured.to_csv(temporary / "imdb_structured.csv", index=False)
    structured.to_csv(temporary / "imdb_structured_joined.csv", index=False)
    joined.to_csv(temporary / "imdb_joined.csv", index=False)

    provenance = {
        "restored_at": datetime.now(timezone.utc).isoformat(),
        "recovered_notebook_commit": NOTEBOOK_COMMIT,
        "review_source": "https://huggingface.co/datasets/fgiobergia/imdb-id",
        "review_relationship": "Stanford IMDb corpus with original movie_id and score restored",
        "review_split": "train",
        "imdb_source": "https://datasets.imdbws.com/",
        "source_sha256": {path.name: sha256(path) for path in (reviews_path, basics_path, crew_path, names_path)},
        "rows": {
            "reviews": len(reviews), "structured_movies": len(structured), "suql_joined": len(joined),
        },
        "unique_review_pairs": int(reviews[["tconst", "review"]].drop_duplicates().shape[0]),
        "notes": "IMDb TSV files are the download-time snapshot; the lost historical snapshot was gitignored.",
    }
    (temporary / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for path in temporary.iterdir():
        os.replace(path, output / path.name)
    temporary.rmdir()
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
