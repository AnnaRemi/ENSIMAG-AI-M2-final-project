#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parent
SOURCE = LAB_ROOT / "project SUQL" / "data" / "imdb_joined.csv"
NEGATIVE_SPLIT_END = 12_500
ROWS_PER_QUESTION = 60
OUTPUT_COLUMNS = [
    "movie_id",
    "title",
    "director",
    "year",
    "runtime",
    "genres",
    "review",
]


@dataclass(frozen=True)
class QuestionSpec:
    number: int
    slug: str
    difficulty: int
    difficulty_label: str
    question: str
    semantic_question: str
    suql_query: str
    structured_filters: list[dict[str, str]]
    semantic_task: str
    target_sentiment: str
    genre: str | None = None
    year: int | None = None
    runtime_op: str | None = None
    runtime_value: int | None = None


QUESTIONS = [
    QuestionSpec(
        number=1,
        slug="question_01_year_2001_negative",
        difficulty=1,
        difficulty_label="easy",
        question="Which movies released in 2001 have reviews expressing an overall negative opinion?",
        semantic_question="Does this review express an overall negative opinion of the movie?",
        suql_query=(
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE year = 2001 AND answer(review, "
            "'Does this review express an overall negative opinion of the movie?') = 'Yes';"
        ),
        structured_filters=[{"column": "year", "op": "eq", "value": "2001"}],
        semantic_task="overall negative sentiment classification",
        target_sentiment="negative",
        year=2001,
    ),
    QuestionSpec(
        number=2,
        slug="question_02_drama_under_100_positive",
        difficulty=2,
        difficulty_label="medium",
        question="Which Drama movies under 100 minutes have reviews expressing an overall positive opinion?",
        semantic_question="Does this review express an overall positive opinion of the movie?",
        suql_query=(
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Drama%' AND runtime < 100 AND answer(review, "
            "'Does this review express an overall positive opinion of the movie?') = 'Yes';"
        ),
        structured_filters=[
            {"column": "genres", "op": "contains", "value": "Drama"},
            {"column": "runtime", "op": "lt", "value": "100"},
        ],
        semantic_task="overall positive sentiment classification",
        target_sentiment="positive",
        genre="Drama",
        runtime_op="lt",
        runtime_value=100,
    ),
    QuestionSpec(
        number=3,
        slug="question_03_comedy_over_90_negative",
        difficulty=2,
        difficulty_label="medium",
        question="Which Comedy movies over 90 minutes have reviews expressing an overall negative opinion?",
        semantic_question="Does this review express an overall negative opinion of the movie?",
        suql_query=(
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Comedy%' AND runtime > 90 AND answer(review, "
            "'Does this review express an overall negative opinion of the movie?') = 'Yes';"
        ),
        structured_filters=[
            {"column": "genres", "op": "contains", "value": "Comedy"},
            {"column": "runtime", "op": "gt", "value": "90"},
        ],
        semantic_task="overall negative sentiment classification",
        target_sentiment="negative",
        genre="Comedy",
        runtime_op="gt",
        runtime_value=90,
    ),
    QuestionSpec(
        number=4,
        slug="question_04_horror_under_100_positive",
        difficulty=3,
        difficulty_label="medium_hard",
        question="Which Horror movies under 100 minutes have reviews expressing an overall positive opinion?",
        semantic_question="Does this review express an overall positive opinion of the movie?",
        suql_query=(
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Horror%' AND runtime < 100 AND answer(review, "
            "'Does this review express an overall positive opinion of the movie?') = 'Yes';"
        ),
        structured_filters=[
            {"column": "genres", "op": "contains", "value": "Horror"},
            {"column": "runtime", "op": "lt", "value": "100"},
        ],
        semantic_task="overall positive sentiment classification",
        target_sentiment="positive",
        genre="Horror",
        runtime_op="lt",
        runtime_value=100,
    ),
    QuestionSpec(
        number=5,
        slug="question_05_action_over_100_negative",
        difficulty=3,
        difficulty_label="medium_hard",
        question="Which Action movies over 100 minutes have reviews expressing an overall negative opinion?",
        semantic_question="Does this review express an overall negative opinion of the movie?",
        suql_query=(
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Action%' AND runtime > 100 AND answer(review, "
            "'Does this review express an overall negative opinion of the movie?') = 'Yes';"
        ),
        structured_filters=[
            {"column": "genres", "op": "contains", "value": "Action"},
            {"column": "runtime", "op": "gt", "value": "100"},
        ],
        semantic_task="overall negative sentiment classification",
        target_sentiment="negative",
        genre="Action",
        runtime_op="gt",
        runtime_value=100,
    ),
    QuestionSpec(
        number=6,
        slug="question_06_romance_under_100_positive",
        difficulty=3,
        difficulty_label="medium_hard",
        question="Which Romance movies under 100 minutes have reviews expressing an overall positive opinion?",
        semantic_question="Does this review express an overall positive opinion of the movie?",
        suql_query=(
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Romance%' AND runtime < 100 AND answer(review, "
            "'Does this review express an overall positive opinion of the movie?') = 'Yes';"
        ),
        structured_filters=[
            {"column": "genres", "op": "contains", "value": "Romance"},
            {"column": "runtime", "op": "lt", "value": "100"},
        ],
        semantic_task="overall positive sentiment classification",
        target_sentiment="positive",
        genre="Romance",
        runtime_op="lt",
        runtime_value=100,
    ),
    QuestionSpec(
        number=7,
        slug="question_07_thriller_under_120_negative",
        difficulty=4,
        difficulty_label="hard",
        question="Which Thriller movies under 120 minutes have reviews expressing an overall negative opinion?",
        semantic_question="Does this review express an overall negative opinion of the movie?",
        suql_query=(
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Thriller%' AND runtime < 120 AND answer(review, "
            "'Does this review express an overall negative opinion of the movie?') = 'Yes';"
        ),
        structured_filters=[
            {"column": "genres", "op": "contains", "value": "Thriller"},
            {"column": "runtime", "op": "lt", "value": "120"},
        ],
        semantic_task="overall negative sentiment classification",
        target_sentiment="negative",
        genre="Thriller",
        runtime_op="lt",
        runtime_value=120,
    ),
    QuestionSpec(
        number=8,
        slug="question_08_crime_over_90_positive",
        difficulty=4,
        difficulty_label="hard",
        question="Which Crime movies over 90 minutes have reviews expressing an overall positive opinion?",
        semantic_question="Does this review express an overall positive opinion of the movie?",
        suql_query=(
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Crime%' AND runtime > 90 AND answer(review, "
            "'Does this review express an overall positive opinion of the movie?') = 'Yes';"
        ),
        structured_filters=[
            {"column": "genres", "op": "contains", "value": "Crime"},
            {"column": "runtime", "op": "gt", "value": "90"},
        ],
        semantic_task="overall positive sentiment classification",
        target_sentiment="positive",
        genre="Crime",
        runtime_op="gt",
        runtime_value=90,
    ),
    QuestionSpec(
        number=9,
        slug="question_09_scifi_over_90_negative",
        difficulty=5,
        difficulty_label="very_hard",
        question="Which Sci-Fi movies over 90 minutes have reviews expressing an overall negative opinion?",
        semantic_question="Does this review express an overall negative opinion of the movie?",
        suql_query=(
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Sci-Fi%' AND runtime > 90 AND answer(review, "
            "'Does this review express an overall negative opinion of the movie?') = 'Yes';"
        ),
        structured_filters=[
            {"column": "genres", "op": "contains", "value": "Sci-Fi"},
            {"column": "runtime", "op": "gt", "value": "90"},
        ],
        semantic_task="overall negative sentiment classification",
        target_sentiment="negative",
        genre="Sci-Fi",
        runtime_op="gt",
        runtime_value=90,
    ),
    QuestionSpec(
        number=10,
        slug="question_10_adventure_under_100_positive",
        difficulty=5,
        difficulty_label="very_hard",
        question="Which Adventure movies under 100 minutes have reviews expressing an overall positive opinion?",
        semantic_question="Does this review express an overall positive opinion of the movie?",
        suql_query=(
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Adventure%' AND runtime < 100 AND answer(review, "
            "'Does this review express an overall positive opinion of the movie?') = 'Yes';"
        ),
        structured_filters=[
            {"column": "genres", "op": "contains", "value": "Adventure"},
            {"column": "runtime", "op": "lt", "value": "100"},
        ],
        semantic_task="overall positive sentiment classification",
        target_sentiment="positive",
        genre="Adventure",
        runtime_op="lt",
        runtime_value=100,
    ),
]


def load_source() -> pd.DataFrame:
    source = pd.read_csv(SOURCE).reset_index(names="source_row")
    source = source.dropna(
        subset=["movie_id", "title", "year", "runtime", "genres", "review"]
    ).copy()
    source["year"] = pd.to_numeric(source["year"], errors="coerce")
    source["runtime"] = pd.to_numeric(source["runtime"], errors="coerce")
    source = source.dropna(subset=["year", "runtime"]).copy()
    source["year"] = source["year"].astype(int)
    source["runtime"] = source["runtime"].astype(int)
    source["source_sentiment"] = source["source_row"].lt(NEGATIVE_SPLIT_END).map(
        {True: "negative", False: "positive"}
    )
    return source


def structured_mask(source: pd.DataFrame, spec: QuestionSpec) -> pd.Series:
    mask = pd.Series(True, index=source.index)
    if spec.genre:
        mask &= source["genres"].str.contains(
            rf"\b{re.escape(spec.genre)}\b",
            case=False,
            na=False,
        )
    if spec.year is not None:
        mask &= source["year"].eq(spec.year)
    if spec.runtime_op == "lt":
        mask &= source["runtime"].lt(spec.runtime_value)
    elif spec.runtime_op == "gt":
        mask &= source["runtime"].gt(spec.runtime_value)
    return mask


def take_unique(
    frame: pd.DataFrame,
    count: int,
    used_movie_ids: set[str],
) -> pd.DataFrame:
    rows = []
    for row in frame.sort_values("source_row").to_dict("records"):
        movie_id = str(row["movie_id"])
        if movie_id in used_movie_ids:
            continue
        rows.append(row)
        used_movie_ids.add(movie_id)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"Expected {count} unique rows, found {len(rows)}")
    return pd.DataFrame(rows)


def write_question(
    spec: QuestionSpec,
    source: pd.DataFrame,
    global_used: set[str],
) -> dict:
    mask = structured_mask(source, spec)
    target = source["source_sentiment"].eq(spec.target_sentiment)
    used = global_used
    selected = pd.concat(
        [
            take_unique(source[mask & target], 12, used),
            take_unique(source[mask & ~target], 12, used),
            take_unique(source[~mask & target], 18, used),
            take_unique(source[~mask & ~target], 18, used),
        ],
        ignore_index=True,
    )
    selected["structured_match"] = structured_mask(selected, spec)
    selected["semantic_label"] = selected["source_sentiment"].eq(spec.target_sentiment)
    selected["ground_truth"] = (
        selected["structured_match"] & selected["semantic_label"]
    ).astype(int)

    question_dir = ROOT / spec.slug
    data_dir = question_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    selected = selected.sort_values(["structured_match", "movie_id"], ascending=[False, True])
    selected[OUTPUT_COLUMNS].to_csv(data_dir / "imdb_joined.csv", index=False)
    selected[["movie_id", "title", "director", "year", "runtime", "genres"]].to_csv(
        data_dir / "imdb_structured_joined.csv",
        index=False,
    )
    selected[["movie_id", "review"]].rename(columns={"movie_id": "tconst"}).to_csv(
        data_dir / "imdb_reviews.csv",
        index=False,
    )

    annotations = selected[
        [
            "movie_id",
            "source_row",
            "title",
            "year",
            "runtime",
            "genres",
            "source_sentiment",
            "structured_match",
            "semantic_label",
            "ground_truth",
        ]
    ].copy()
    annotations["annotation_source"] = "IMDb 50K source sentiment split"
    annotations["evidence_excerpt"] = ""
    annotations["annotation_rationale"] = annotations.apply(
        lambda row: (
            f"Included: structured filters match and source sentiment is {spec.target_sentiment}."
            if row["ground_truth"]
            else (
                "Excluded by structured filters."
                if not row["structured_match"]
                else f"Excluded by semantic label: source sentiment is not {spec.target_sentiment}."
            )
        ),
        axis=1,
    )
    annotations.to_csv(data_dir / "annotations.csv", index=False)
    annotations[annotations["ground_truth"].eq(1)][
        ["movie_id", "title", "ground_truth", "annotation_rationale", "evidence_excerpt"]
    ].to_csv(data_dir / "ground_truth.csv", index=False)

    ground_truth_ids = sorted(
        selected.loc[selected["ground_truth"].eq(1), "movie_id"].astype(str)
    )
    if not ground_truth_ids:
        raise RuntimeError(f"{spec.slug} has empty ground truth")
    benchmark = {
        "benchmark_id": f"common_benchmark_10q_q{spec.number:02d}",
        "difficulty": spec.difficulty,
        "difficulty_label": spec.difficulty_label,
        "question": spec.question,
        "semantic_question": spec.semantic_question,
        "suql_query": spec.suql_query,
        "structured_filters": spec.structured_filters,
        "semantic_task": spec.semantic_task,
        "annotation_policy": "IMDb source sentiment label; no evaluated LLM used",
        "row_count": ROWS_PER_QUESTION,
        "structured_candidate_count": int(selected["structured_match"].sum()),
        "ground_truth_movie_ids": ground_truth_ids,
    }
    (question_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    return benchmark


def main() -> None:
    source = load_source()
    global_used: set[str] = set()
    build_order = [6, 10, 1, 2, 3, 4, 5, 7, 8, 9]
    by_number = {item.number: item for item in QUESTIONS}
    built = {
        number: write_question(by_number[number], source, global_used)
        for number in build_order
    }
    manifest = {
        "benchmark_id": "common_benchmark_10q",
        "source": str(SOURCE.relative_to(LAB_ROOT)),
        "questions": [
            {
                "directory": by_number[number].slug,
                "benchmark_id": built[number]["benchmark_id"],
                "difficulty": built[number]["difficulty"],
                "question": built[number]["question"],
                "row_count": built[number]["row_count"],
                "structured_candidate_count": built[number]["structured_candidate_count"],
                "ground_truth_count": len(built[number]["ground_truth_movie_ids"]),
            }
            for number in sorted(built)
        ],
        "dataset_policy": (
            "Each question contains 60 unique movie IDs with one review per movie. "
            "Movie IDs are disjoint across questions. Ground truth is the intersection "
            "of deterministic structured filters and the IMDb source sentiment split."
        ),
    }
    if any(item["ground_truth_count"] <= 0 for item in manifest["questions"]):
        raise RuntimeError("Every question must have non-empty ground truth")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
