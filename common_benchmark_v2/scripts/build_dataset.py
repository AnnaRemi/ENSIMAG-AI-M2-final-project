#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parent
SOURCE = LAB_ROOT / "project SUQL" / "data" / "imdb_joined.csv"
DATA_DIR = ROOT / "data"

QUESTION = (
    "Which movies released in 1998 have reviews expressing an overall negative, "
    "critical, or strongly unfavorable opinion of the movie?"
)
SEMANTIC_QUESTION = (
    "Does this review express an overall negative, critical, or strongly unfavorable "
    "opinion of the movie?"
)
SUQL = (
    "SELECT movie_id, title, year, runtime, director, genres "
    "FROM movies "
    "WHERE year = 1998 "
    f"AND answer(review, '{SEMANTIC_QUESTION}') = 'Yes';"
)

# project SUQL/data/imdb_joined.csv preserves the ordering of the source
# IMDb 50K test split: rows 0..12499 are negative and 12500..24999 positive.
NEGATIVE_SPLIT_END = 12_500
TARGET_YEAR = 1998
TARGET_NEGATIVE = 13
TARGET_POSITIVE = 12
OTHER_NEGATIVE = 13
OTHER_POSITIVE = 12


def take_first(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    result = frame.sort_values("source_row").head(count).copy()
    if len(result) != count:
        raise RuntimeError(f"Expected {count} rows, found {len(result)}")
    return result


def take_year_diverse(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Round-robin over years so non-1998 distractors are not one-year-only."""

    groups = {
        int(year): group.sort_values("source_row").to_dict("records")
        for year, group in frame.groupby("year")
    }
    years = sorted(groups, key=lambda year: (-len(groups[year]), year))
    selected: list[dict] = []
    while len(selected) < count:
        progress = False
        for year in years:
            if groups[year] and len(selected) < count:
                selected.append(groups[year].pop(0))
                progress = True
        if not progress:
            break
    if len(selected) != count:
        raise RuntimeError(f"Expected {count} diverse-year rows, found {len(selected)}")
    return pd.DataFrame(selected)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE).reset_index(names="source_row")
    source = source.dropna(subset=["movie_id", "title", "year", "review"]).copy()
    source["year"] = pd.to_numeric(source["year"], errors="coerce")
    source = source.dropna(subset=["year"])
    source["year"] = source["year"].astype(int)
    source["source_sentiment"] = source["source_row"].lt(NEGATIVE_SPLIT_END).map(
        {True: "negative", False: "positive"}
    )

    # Keep one review per movie so both systems operate on the same 50 unique IDs.
    source = source.sort_values("source_row").drop_duplicates("movie_id", keep="first")

    target = source[source["year"].eq(TARGET_YEAR)]
    other = source[~source["year"].eq(TARGET_YEAR)]
    selected = pd.concat(
        [
            take_first(target[target["source_sentiment"].eq("negative")], TARGET_NEGATIVE),
            take_first(target[target["source_sentiment"].eq("positive")], TARGET_POSITIVE),
            take_year_diverse(other[other["source_sentiment"].eq("negative")], OTHER_NEGATIVE),
            take_year_diverse(other[other["source_sentiment"].eq("positive")], OTHER_POSITIVE),
        ],
        ignore_index=True,
    )

    selected["ground_truth"] = (
        selected["year"].eq(TARGET_YEAR)
        & selected["source_sentiment"].eq("negative")
    ).astype(int)
    selected["category"] = selected.apply(
        lambda row: (
            f"{row['year']}_{row['source_sentiment']}"
            if row["year"] == TARGET_YEAR
            else f"other_year_{row['source_sentiment']}"
        ),
        axis=1,
    )
    selected["annotation_rationale"] = selected.apply(
        lambda row: (
            "Ground truth: movie year is 1998 and the source IMDb sentiment label is negative."
            if row["ground_truth"]
            else (
                "Excluded by sentiment: movie year is 1998 but the source IMDb sentiment label is positive."
                if row["year"] == TARGET_YEAR
                else f"Excluded by year: movie year is {row['year']}, not 1998."
            )
        ),
        axis=1,
    )
    selected = selected.sort_values(
        ["year", "source_sentiment", "movie_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    joined_columns = ["movie_id", "title", "director", "year", "runtime", "genres", "review"]
    selected[joined_columns].to_csv(DATA_DIR / "imdb_joined.csv", index=False)
    selected[
        ["movie_id", "title", "director", "year", "runtime", "genres"]
    ].to_csv(DATA_DIR / "imdb_structured_joined.csv", index=False)
    selected[["movie_id", "review"]].rename(columns={"movie_id": "tconst"}).to_csv(
        DATA_DIR / "imdb_reviews.csv", index=False
    )
    selected[
        [
            "source_row",
            "movie_id",
            "title",
            "year",
            "source_sentiment",
            "ground_truth",
            "category",
            "annotation_rationale",
        ]
    ].to_csv(DATA_DIR / "annotations.csv", index=False)

    ground_truth_ids = sorted(
        selected.loc[selected["ground_truth"].eq(1), "movie_id"].astype(str).unique()
    )
    represented_years = sorted(selected["year"].astype(int).unique().tolist())
    benchmark = {
        "benchmark_id": "negative_reviews_1998_v2_50_rows",
        "question": QUESTION,
        "semantic_question": SEMANTIC_QUESTION,
        "suql_query": SUQL,
        "trummer_join_predicate": (
            "the movie row has year=1998, the review chunk is about the same movie "
            "based on movie_id/tconst, and the review expresses an overall negative, "
            "critical, or strongly unfavorable opinion of the movie"
        ),
        "ground_truth_movie_ids": ground_truth_ids,
        "row_count": int(len(selected)),
        "candidate_1998_rows": int(selected["year"].eq(TARGET_YEAR).sum()),
        "ground_truth_count": len(ground_truth_ids),
        "represented_years": represented_years,
        "annotation_unit": "movie_id",
        "data_policy": (
            "50 unique movies with one review each; source sentiment comes from the "
            "IMDb 50K test-split ordering; ground truth is year=1998 AND negative"
        ),
        "trummer_year_policy": "year is evaluated inside the semantic join predicate",
    }
    (ROOT / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")

    assert len(selected) == 50
    assert selected["movie_id"].nunique() == 50
    assert selected["year"].eq(TARGET_YEAR).sum() == 25
    assert len(ground_truth_ids) == TARGET_NEGATIVE
    assert len(represented_years) > 2
    print(
        f"Wrote 50 rows across {len(represented_years)} years, "
        f"25 rows from 1998, and {len(ground_truth_ids)} ground-truth IDs"
    )


if __name__ == "__main__":
    main()
