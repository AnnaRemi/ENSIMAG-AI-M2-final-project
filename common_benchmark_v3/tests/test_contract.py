from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkContractTests(unittest.TestCase):
    def test_dataset_and_ground_truth_contract(self) -> None:
        spec = json.loads((ROOT / "benchmark.json").read_text())
        with (ROOT / "data" / "imdb_structured_joined.csv").open(newline="", encoding="utf-8") as handle:
            movies = list(csv.DictReader(handle))
        with (ROOT / "data" / "imdb_reviews.csv").open(newline="", encoding="utf-8") as handle:
            reviews = list(csv.DictReader(handle))
        self.assertEqual(len(movies), 50)
        self.assertEqual(len(reviews), 50)
        self.assertEqual(len(spec["ground_truth_movie_ids"]), 13)
        self.assertEqual(
            set(spec["implementations"]),
            {"trummer_heterogen_v1", "trummer_heterogen_v2_cascade"},
        )


if __name__ == "__main__":
    unittest.main()
