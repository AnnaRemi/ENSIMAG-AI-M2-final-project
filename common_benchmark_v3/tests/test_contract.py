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
            {
                "suql_baseline",
                "trummer_heterogen_v1",
                "trummer_heterogen_v2_2_structured_pruned",
                "trummer_heterogen_v2_3_batched_cascade",
                "trummer_heterogen_v3_pruned_cascade",
                "trummer_heterogen_v2_cascade",
            },
        )
        self.assertIn("suql_query", spec)

    def test_all_heterogen_aker_workflow_files_exist(self) -> None:
        scripts = ROOT / "scripts"
        required = {
            "run_all_heterogen.py",
            "evaluate_all_heterogen.py",
            "sync_all_heterogen_to_aker.sh",
            "submit_aker_all_heterogen.sh",
            "run_aker_all_heterogen.sh",
            "pull_all_heterogen_from_aker.sh",
        }
        self.assertTrue(
            required <= {path.name for path in scripts.iterdir()},
        )


if __name__ == "__main__":
    unittest.main()
