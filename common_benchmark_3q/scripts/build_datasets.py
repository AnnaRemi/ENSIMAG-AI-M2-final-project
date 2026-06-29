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
class ManualLabel:
    source_row: int
    ground_truth: int
    evidence_marker: str
    rationale: str


Q2_LABELS = [
    ManualLabel(12542, 1, "best performances", "Explicitly praises Dana Andrews' performance."),
    ManualLabel(12580, 1, "capital performance", "Explicitly praises the lead and supporting performances."),
    ManualLabel(12601, 1, "great acting", "Directly describes the acting as great."),
    ManualLabel(12745, 1, "perfect performances", "Directly praises all actors' performances."),
    ManualLabel(12811, 1, "whole cast", "Calls the whole cast superb."),
    ManualLabel(13001, 1, "acting is largely very good", "Directly praises the acting and lead performance."),
    ManualLabel(13029, 1, "acting unblemished", "Directly describes the acting as unblemished."),
    ManualLabel(13156, 1, "best acting", "Explicitly praises multiple performances."),
    ManualLabel(13498, 1, "best performances", "Calls Bill Nighy's work one of the best performances."),
    ManualLabel(13783, 1, "good performances", "Directly describes multiple performances as good."),
    ManualLabel(15, 0, "acting ability", "Says meaningful acting ability is absent."),
    ManualLabel(18, 0, "acting was", "Describes most acting as amateur."),
    ManualLabel(22, 0, "acting", "Describes the acting as weak."),
    ManualLabel(89, 0, "lead male actor", "Criticizes both lead performers."),
    ManualLabel(136, 0, "performances", "Compares performances to an elementary-school pageant."),
    ManualLabel(142, 0, "worse acting", "Directly criticizes the acting."),
    ManualLabel(181, 0, "acting is appalling", "Directly describes the acting as appalling."),
    ManualLabel(262, 0, "bad acting", "Directly describes the acting as bad."),
    ManualLabel(276, 0, "acting was my main gripe", "Directly describes the acting as awful."),
    ManualLabel(281, 0, "wooden performance", "Directly criticizes the lead performance."),
]


Q3_LABELS = [
    ManualLabel(11, 1, "special effects", "Overall negative, but explicitly praises the special effects."),
    ManualLabel(26, 1, "performances were good", "Overall negative, but explicitly praises several performances."),
    ManualLabel(133, 1, "save this poor movie", "Overall negative, but explicitly praises Annie Potts."),
    ManualLabel(350, 1, "very good actor", "Calls the movie a waste but praises Anthony Quinn."),
    ManualLabel(767, 1, "strong cast", "Calls the film overrated while praising cast and dialogue."),
    ManualLabel(891, 1, "Peter Falk is great", "Calls the movie weak while explicitly praising Peter Falk."),
    ManualLabel(943, 1, "performance was wonderful", "Overall negative, but praises one performance as wonderful."),
    ManualLabel(1286, 1, "acting was very good", "Calls it the worst movie while explicitly praising the acting."),
    ManualLabel(7021, 1, "actors were very good", "Calls the movie a failure while praising direction and actors."),
    ManualLabel(7399, 1, "amazing acting", "Rates the movie 2/10 while explicitly praising Al Pacino."),
    ManualLabel(58, 0, "not a single good performance", "Overall negative and explicitly rejects any good performance."),
    ManualLabel(517, 0, "pure guano", "Overall negative with no explicit praise of a movie aspect."),
    ManualLabel(570, 0, "annoying, obnoxious", "Overall negative with no explicit praise of a movie aspect."),
    ManualLabel(689, 0, "crap", "Overall negative with no explicit praise of a movie aspect."),
    ManualLabel(700, 0, "horrible effort", "Mentions notable actors but does not praise their work in this movie."),
    ManualLabel(1002, 0, "thoroughly unpleasant", "Overall negative with no explicit praise of execution."),
    ManualLabel(1247, 0, "doesn't work", "Overall negative and says the acting cannot save the movie."),
    ManualLabel(1331, 0, "bad acting", "Overall negative with no praised aspect."),
    ManualLabel(1382, 0, "sad waste", "Overall negative and calls the cast wasted rather than praised."),
    ManualLabel(1942, 0, "puerile and unfunny", "Overall negative with no explicit praise of a specific aspect."),
    ManualLabel(12646, 0, "cast is great", "Praises an aspect, but the review is overall positive."),
    ManualLabel(12711, 0, "superb direction", "Praises aspects, but the review is overall positive."),
    ManualLabel(13387, 0, "actors are great", "Praises aspects, but the review is overall positive."),
    ManualLabel(13984, 0, "good laughs", "Praises aspects, but the review is overall positive."),
    ManualLabel(14072, 0, "best actor", "Praises an aspect, but the review is overall positive."),
    ManualLabel(14286, 0, "performances are top notch", "Praises aspects, but the review is overall positive."),
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


def rows_by_source(source: pd.DataFrame, labels: list[ManualLabel]) -> pd.DataFrame:
    indexed = source.set_index("source_row", drop=False)
    missing = [item.source_row for item in labels if item.source_row not in indexed.index]
    if missing:
        raise RuntimeError(f"Missing curated source rows: {missing}")
    return pd.DataFrame([indexed.loc[item.source_row].to_dict() for item in labels])


def evidence_excerpt(review: str, marker: str, radius: int = 120) -> str:
    normalized = " ".join(str(review).replace("<br />", " ").split())
    match = re.search(re.escape(marker), normalized, re.IGNORECASE)
    if not match:
        raise RuntimeError(f"Evidence marker not found: {marker!r}")
    start = max(0, match.start() - radius)
    end = min(len(normalized), match.end() + radius)
    return normalized[start:end]


def write_question(
    slug: str,
    frame: pd.DataFrame,
    annotations: pd.DataFrame,
    spec: dict,
) -> None:
    question_dir = ROOT / slug
    data_dir = question_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(["structured_match", "movie_id"], ascending=[False, True])
    frame[OUTPUT_COLUMNS].to_csv(data_dir / "imdb_joined.csv", index=False)
    frame[
        ["movie_id", "title", "director", "year", "runtime", "genres"]
    ].to_csv(data_dir / "imdb_structured_joined.csv", index=False)
    frame[["movie_id", "review"]].rename(columns={"movie_id": "tconst"}).to_csv(
        data_dir / "imdb_reviews.csv",
        index=False,
    )
    annotations = annotations.set_index("movie_id").loc[frame["movie_id"]].reset_index()
    annotations.to_csv(data_dir / "annotations.csv", index=False)
    annotations[annotations["ground_truth"].eq(1)][
        ["movie_id", "title", "ground_truth", "annotation_rationale", "evidence_excerpt"]
    ].to_csv(data_dir / "ground_truth.csv", index=False)
    (question_dir / "benchmark.json").write_text(json.dumps(spec, indent=2) + "\n")


def build_easy(source: pd.DataFrame, used: set[str]) -> dict:
    target = source[source["year"].eq(2001)]
    selected = pd.concat(
        [
            take_unique(target[target["source_sentiment"].eq("negative")], 12, used),
            take_unique(target[target["source_sentiment"].eq("positive")], 12, used),
            take_unique(
                source[
                    ~source["year"].eq(2001)
                    & source["source_sentiment"].eq("negative")
                ],
                18,
                used,
            ),
            take_unique(
                source[
                    ~source["year"].eq(2001)
                    & source["source_sentiment"].eq("positive")
                ],
                18,
                used,
            ),
        ],
        ignore_index=True,
    )
    selected["structured_match"] = selected["year"].eq(2001)
    selected["semantic_label"] = selected["source_sentiment"].eq("negative")
    selected["ground_truth"] = (
        selected["structured_match"] & selected["semantic_label"]
    ).astype(int)
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
            "Included: year is 2001 and the source sentiment label is negative."
            if row["ground_truth"]
            else (
                "Excluded by year."
                if not row["structured_match"]
                else "Excluded by semantic label: source sentiment is positive."
            )
        ),
        axis=1,
    )
    spec = {
        "benchmark_id": "common_benchmark_three_questions_q1_easy",
        "difficulty": 1,
        "difficulty_label": "easy",
        "question": "Which movies released in 2001 have reviews expressing an overall negative opinion?",
        "semantic_question": "Does this review express an overall negative opinion of the movie?",
        "suql_query": (
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE year = 2001 AND answer(review, "
            "'Does this review express an overall negative opinion of the movie?') = 'Yes';"
        ),
        "structured_filters": [{"column": "year", "op": "eq", "value": "2001"}],
        "semantic_task": "overall sentiment classification",
        "annotation_policy": "IMDb source sentiment label; no evaluated LLM used",
        "row_count": 60,
        "structured_candidate_count": 24,
        "ground_truth_movie_ids": sorted(
            selected.loc[selected["ground_truth"].eq(1), "movie_id"].astype(str)
        ),
    }
    write_question("question_1_easy", selected, annotations, spec)
    return spec


def build_medium(source: pd.DataFrame, used: set[str]) -> dict:
    labels = Q2_LABELS
    curated = rows_by_source(source, labels)
    is_drama = source["genres"].str.contains(r"\bDrama\b", case=False, na=False)
    wrong_genre = take_unique(
        source[~is_drama & source["runtime"].lt(100)],
        20,
        used,
    )
    wrong_runtime = take_unique(
        source[is_drama & source["runtime"].ge(100)],
        20,
        used,
    )
    selected = pd.concat([curated, wrong_genre, wrong_runtime], ignore_index=True)
    selected["structured_match"] = (
        selected["genres"].str.contains(r"\bDrama\b", case=False, na=False)
        & selected["runtime"].lt(100)
    )
    labels_by_row = {item.source_row: item for item in labels}
    selected["semantic_label"] = selected["source_row"].map(
        lambda value: bool(labels_by_row.get(int(value), ManualLabel(0, 0, "", "")).ground_truth)
    )
    selected["ground_truth"] = (
        selected["structured_match"] & selected["semantic_label"]
    ).astype(int)
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
    annotations["annotation_source"] = annotations["source_row"].map(
        lambda value: "manual evidence annotation" if int(value) in labels_by_row else "structured distractor"
    )
    annotations["evidence_excerpt"] = selected.apply(
        lambda row: (
            evidence_excerpt(
                row["review"],
                labels_by_row[int(row["source_row"])].evidence_marker,
            )
            if int(row["source_row"]) in labels_by_row
            else ""
        ),
        axis=1,
    )
    annotations["annotation_rationale"] = selected.apply(
        lambda row: (
            labels_by_row[int(row["source_row"])].rationale
            if int(row["source_row"]) in labels_by_row
            else "Excluded by structured filters."
        ),
        axis=1,
    )
    spec = {
        "benchmark_id": "common_benchmark_three_questions_q2_medium",
        "difficulty": 2,
        "difficulty_label": "medium",
        "question": "Which Drama movies under 100 minutes have reviews that explicitly praise the acting or performances?",
        "semantic_question": "Does this review explicitly praise the acting, cast, portrayal, or performances in this movie?",
        "suql_query": (
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Drama%' AND runtime < 100 AND answer(review, "
            "'Does this review explicitly praise the acting, cast, portrayal, or performances in this movie?') = 'Yes';"
        ),
        "structured_filters": [
            {"column": "genres", "op": "contains", "value": "Drama"},
            {"column": "runtime", "op": "lt", "value": "100"},
        ],
        "semantic_task": "aspect-level praise extraction",
        "annotation_policy": "manually curated evidence excerpts; no evaluated LLM used",
        "row_count": 60,
        "structured_candidate_count": 20,
        "ground_truth_movie_ids": sorted(
            selected.loc[selected["ground_truth"].eq(1), "movie_id"].astype(str)
        ),
    }
    write_question("question_2_medium", selected, annotations, spec)
    return spec


def build_hard(source: pd.DataFrame, used: set[str]) -> dict:
    labels = Q3_LABELS
    curated = rows_by_source(source, labels)
    is_comedy = source["genres"].str.contains(r"\bComedy\b", case=False, na=False)
    wrong_genre = take_unique(
        source[~is_comedy & source["runtime"].gt(90)],
        17,
        used,
    )
    wrong_runtime = take_unique(
        source[is_comedy & source["runtime"].le(90)],
        17,
        used,
    )
    selected = pd.concat([curated, wrong_genre, wrong_runtime], ignore_index=True)
    selected["structured_match"] = (
        selected["genres"].str.contains(r"\bComedy\b", case=False, na=False)
        & selected["runtime"].gt(90)
    )
    labels_by_row = {item.source_row: item for item in labels}
    selected["semantic_label"] = selected["source_row"].map(
        lambda value: bool(labels_by_row.get(int(value), ManualLabel(0, 0, "", "")).ground_truth)
    )
    selected["ground_truth"] = (
        selected["structured_match"] & selected["semantic_label"]
    ).astype(int)
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
    annotations["annotation_source"] = annotations["source_row"].map(
        lambda value: "manual contrastive annotation" if int(value) in labels_by_row else "structured distractor"
    )
    annotations["evidence_excerpt"] = selected.apply(
        lambda row: (
            evidence_excerpt(
                row["review"],
                labels_by_row[int(row["source_row"])].evidence_marker,
            )
            if int(row["source_row"]) in labels_by_row
            else ""
        ),
        axis=1,
    )
    annotations["annotation_rationale"] = selected.apply(
        lambda row: (
            labels_by_row[int(row["source_row"])].rationale
            if int(row["source_row"]) in labels_by_row
            else "Excluded by structured filters."
        ),
        axis=1,
    )
    spec = {
        "benchmark_id": "common_benchmark_three_questions_q3_hard",
        "difficulty": 3,
        "difficulty_label": "hard",
        "question": (
            "Which Comedy movies over 90 minutes have reviews that are overall unfavorable "
            "but still explicitly praise at least one specific aspect such as acting, "
            "direction, soundtrack, visuals, or effects?"
        ),
        "semantic_question": (
            "Is this review overall unfavorable toward the movie while still explicitly "
            "praising at least one specific aspect of this movie?"
        ),
        "suql_query": (
            "SELECT movie_id, title, year, runtime, director, genres FROM movies "
            "WHERE genres LIKE '%Comedy%' AND runtime > 90 AND answer(review, "
            "'Is this review overall unfavorable toward the movie while still explicitly "
            "praising at least one specific aspect of this movie?') = 'Yes';"
        ),
        "structured_filters": [
            {"column": "genres", "op": "contains", "value": "Comedy"},
            {"column": "runtime", "op": "gt", "value": "90"},
        ],
        "semantic_task": "contrastive overall sentiment plus aspect-level praise",
        "annotation_policy": "manually curated contrastive evidence; no evaluated LLM used",
        "row_count": 60,
        "structured_candidate_count": len(labels),
        "ground_truth_movie_ids": sorted(
            selected.loc[selected["ground_truth"].eq(1), "movie_id"].astype(str)
        ),
    }
    write_question("question_3_hard", selected, annotations, spec)
    return spec


def main() -> None:
    source = load_source()
    q2_ids = set(rows_by_source(source, Q2_LABELS)["movie_id"].astype(str))
    q3_ids = set(rows_by_source(source, Q3_LABELS)["movie_id"].astype(str))
    overlap = q2_ids & q3_ids
    if overlap:
        raise RuntimeError(f"Curated question IDs overlap: {sorted(overlap)}")
    # Reserve all manually curated movies before selecting any distractors.
    used: set[str] = q2_ids | q3_ids
    specs = [
        build_medium(source, used),
        build_hard(source, used),
        build_easy(source, used),
    ]
    specs.sort(key=lambda item: item["difficulty"])
    manifest = {
        "benchmark_id": "common_benchmark_three_questions",
        "source": str(SOURCE.relative_to(LAB_ROOT)),
        "questions": [
            {
                "directory": f"question_{item['difficulty']}_{item['difficulty_label']}",
                "benchmark_id": item["benchmark_id"],
                "difficulty": item["difficulty"],
                "question": item["question"],
                "row_count": item["row_count"],
                "structured_candidate_count": item["structured_candidate_count"],
                "ground_truth_count": len(item["ground_truth_movie_ids"]),
            }
            for item in specs
        ],
        "dataset_policy": (
            "Each question contains 60 unique movie IDs with one review per movie. "
            "Movie IDs are disjoint across questions. Structured distractors are included "
            "so semantic evaluation requires deterministic pruning first."
        ),
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
