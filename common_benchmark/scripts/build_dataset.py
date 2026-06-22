#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parent
SOURCE = LAB_ROOT / "project SUQL" / "data" / "imdb_joined.csv"
DATA_DIR = ROOT / "data"

# Stable source row numbers from project SUQL/data/imdb_joined.csv.
# Every selected movie has exactly one review in the common benchmark.
SELECTION = [
    (366, 1, "1998_negative", "Calls the film and its acting, directing, and writing bad and boring."),
    (7356, 1, "1998_negative", "Says every major aspect of the film is bad and it has no redeeming qualities."),
    (10254, 1, "1998_negative", "Describes the film as among the worst, with awful effects and script."),
    (6591, 1, "1998_negative", "Explicitly calls the movie awful and criticizes its premise and execution."),
    (4749, 1, "1998_negative", "Calls it one of the worst movies and the acting awful."),
    (11871, 1, "1998_negative", "Repeatedly calls the movie bad and says nothing saves it."),
    (14973, 0, "1998_positive", "Strongly praises the film's charm and entertainment value."),
    (19410, 0, "1998_positive", "Praises the science-fiction story and lead performance."),
    (23672, 0, "1998_positive", "Reports a warm, happy experience and calls the acting superb."),
    (17830, 0, "1998_positive", "Calls it a great film with wonderful acting."),
    (13835, 0, "1998_positive", "Calls it excellent and praises the performances."),
    (15033, 0, "1998_positive", "Says the reviewer absolutely loved the show and calls it wonderful."),
    (3011, 0, "1997_negative_distractor", "Negative review, but the movie is from 1997."),
    (1048, 0, "1997_negative_distractor", "Negative review, but the movie is from 1997."),
    (7179, 0, "1997_negative_distractor", "Negative review, but the movie is from 1997."),
    (11467, 0, "1997_negative_distractor", "Negative review, but the movie is from 1997."),
]

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


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE).reset_index(names="source_row")
    metadata = {
        source_row: (label, category, rationale)
        for source_row, label, category, rationale in SELECTION
    }
    selected = source[source["source_row"].isin(metadata)].copy()

    expected_rows = [row for row, *_ in SELECTION]
    found_rows = selected["source_row"].tolist()
    if sorted(found_rows) != sorted(expected_rows):
        missing = sorted(set(expected_rows) - set(found_rows))
        raise RuntimeError(f"Source data changed; missing source rows: {missing}")
    if selected["movie_id"].duplicated().any():
        duplicates = selected.loc[selected["movie_id"].duplicated(), "movie_id"].tolist()
        raise RuntimeError(f"Benchmark requires one review per movie; duplicates: {duplicates}")

    selected["ground_truth"] = selected["source_row"].map(lambda row: metadata[row][0])
    selected["category"] = selected["source_row"].map(lambda row: metadata[row][1])
    selected["annotation_rationale"] = selected["source_row"].map(lambda row: metadata[row][2])
    selected = selected.sort_values(["year", "ground_truth", "movie_id"], ascending=[False, False, True])

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
            "ground_truth",
            "category",
            "annotation_rationale",
        ]
    ].to_csv(DATA_DIR / "annotations.csv", index=False)

    ground_truth_ids = sorted(
        selected.loc[selected["ground_truth"].eq(1), "movie_id"].astype(str).unique()
    )
    benchmark = {
        "benchmark_id": "negative_reviews_1998_v1",
        "question": QUESTION,
        "semantic_question": SEMANTIC_QUESTION,
        "suql_query": SUQL,
        "ground_truth_movie_ids": ground_truth_ids,
        "row_count": int(len(selected)),
        "candidate_1998_rows": int(selected["year"].eq(1998).sum()),
        "ground_truth_count": len(ground_truth_ids),
        "annotation_unit": "movie_id",
        "data_policy": "one manually annotated review per unique movie_id",
    }
    (ROOT / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")

    assert len(selected) == 16
    assert selected["year"].eq(1998).sum() == 12
    assert len(ground_truth_ids) == 6
    print(f"Wrote {len(selected)} common rows and {len(ground_truth_ids)} ground-truth IDs")


if __name__ == "__main__":
    main()
