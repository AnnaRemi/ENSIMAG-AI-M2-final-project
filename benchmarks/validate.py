#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data" / "subdatasets"
EXPECTED = {"10q": 10, "5q": 5, "3q": 3, "1q": 1}
REQUIRED_DATA = {
    "annotations.csv", "ground_truth.csv", "imdb_joined.csv",
    "imdb_reviews.csv", "imdb_structured_joined.csv",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    for suite_name, expected_count in EXPECTED.items():
        suite = ROOT / suite_name
        manifest = json.loads((suite / "manifest.json").read_text())
        assert manifest["question_count"] == expected_count
        assert len(manifest["questions"]) == expected_count
        assert (suite / "questions.txt").stat().st_size > 0
        assert (suite / "ground_truth_movies.txt").stat().st_size > 0
        semantic_tasks, structured_signatures = set(), set()
        for index, item in enumerate(manifest["questions"], 1):
            assert item["directory"] == f"q_{index:02d}"
            qdir = suite / "per_question" / item["directory"]
            spec = json.loads((qdir / "benchmark.json").read_text())
            data_dir = DATA_ROOT / suite_name / item["directory"]
            assert REQUIRED_DATA <= {path.name for path in data_dir.iterdir()}
            movies = rows(data_dir / "imdb_structured_joined.csv")
            reviews = rows(data_dir / "imdb_reviews.csv")
            truth = rows(data_dir / "ground_truth.csv")
            truth_ids = set(spec["ground_truth_movie_ids"])
            assert len(movies) == len(reviews) == 100
            assert len(truth_ids) >= 10
            assert {row["movie_id"] for row in truth} == truth_ids
            assert truth_ids <= {row["movie_id"] for row in movies}
            assert spec["structured_candidate_count"] >= len(truth_ids)
            semantic_tasks.add(spec["semantic_task"])
            structured_signatures.add(tuple((f["column"], f["op"]) for f in spec["structured_filters"]))
        if expected_count > 1:
            assert len(semantic_tasks) == expected_count
            assert len(structured_signatures) >= min(expected_count, 5)
        print(f"validated {suite_name}: {expected_count} questions")


if __name__ == "__main__":
    main()
