from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HETEROGEN_V3 = ROOT.parent / "project Trummer" / "heterogen_v3"
sys.path.insert(0, str(HETEROGEN_V3))

from trummer_join.structured_filter import extract_structured_filters


class CommonBaselineContractTests(unittest.TestCase):
    def test_question_contracts(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text())
        all_movie_ids: set[str] = set()
        self.assertEqual(len(manifest["questions"]), 3)

        for item in manifest["questions"]:
            question_dir = ROOT / item["directory"]
            spec = json.loads((question_dir / "benchmark.json").read_text())
            with (question_dir / "data" / "imdb_structured_joined.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                movies = list(csv.DictReader(handle))
            with (question_dir / "data" / "imdb_reviews.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                reviews = list(csv.DictReader(handle))
            with (question_dir / "data" / "annotations.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                annotations = list(csv.DictReader(handle))
            with (question_dir / "data" / "ground_truth.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                ground_truth_rows = list(csv.DictReader(handle))

            movie_ids = {row["movie_id"] for row in movies}
            review_ids = {row["tconst"] for row in reviews}
            truth_ids = set(spec["ground_truth_movie_ids"])
            structured_count = sum(
                row["structured_match"].lower() == "true" for row in annotations
            )

            self.assertEqual(len(movies), 60)
            self.assertEqual(len(reviews), 60)
            self.assertEqual(len(annotations), 60)
            self.assertEqual(len(movie_ids), 60)
            self.assertEqual(movie_ids, review_ids)
            self.assertTrue(truth_ids)
            self.assertTrue(truth_ids <= movie_ids)
            self.assertEqual(
                {row["movie_id"] for row in ground_truth_rows},
                truth_ids,
            )
            self.assertEqual(structured_count, spec["structured_candidate_count"])
            self.assertFalse(all_movie_ids & movie_ids)
            all_movie_ids.update(movie_ids)

    def test_questions_extract_expected_structured_filters(self) -> None:
        for question_dir in sorted(ROOT.glob("question_*")):
            spec = json.loads((question_dir / "benchmark.json").read_text())
            with (question_dir / "data" / "imdb_structured_joined.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                columns = next(csv.reader(handle))
            extracted = [
                {"column": item.column, "op": item.op, "value": item.value}
                for item in extract_structured_filters(spec["question"], columns)
            ]
            sort_key = lambda item: (item["column"], item["op"], item["value"])
            self.assertEqual(
                sorted(extracted, key=sort_key),
                sorted(spec["structured_filters"], key=sort_key),
            )


if __name__ == "__main__":
    unittest.main()
