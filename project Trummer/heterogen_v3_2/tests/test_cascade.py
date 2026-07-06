from __future__ import annotations

import unittest

from trummer_join.cascade import (
    CascadeConfig,
    CascadeJoin,
    _parse_batch_scores,
)


class BatchedCascadeTests(unittest.TestCase):
    def test_one_cheap_call_per_batch_and_fewer_expensive_calls(self) -> None:
        movies = [
            {"movie_id": f"tt{index}", "text": f"movie {index}"}
            for index in range(20)
        ]
        reviews = [
            {"tconst": f"tt{index}", "review": "review", "text": "review"}
            for index in range(20)
        ]
        cheap_batches = []
        expensive_batches = []

        def cheap_score(candidates, predicate):
            del predicate
            cheap_batches.append([item.candidate_id for item in candidates])
            return {item.candidate_id: 0.0 for item in candidates}

        def expensive_classify(candidates, predicate):
            del predicate
            expensive_batches.append([item.candidate_id for item in candidates])
            return set()

        join = CascadeJoin(
            CascadeConfig(
                cheap_batch_size=4,
                expensive_batch_size=12,
                calibration_budget=0,
            ),
            cheap_score_batch=cheap_score,
            expensive_classify=expensive_classify,
        )
        _, _, metrics = join.run(movies, reviews, "negative")

        self.assertEqual(metrics.cheap_calls, 5)
        self.assertEqual(metrics.cheap_batches, 5)
        self.assertEqual(metrics.expensive_calls, 2)
        self.assertLess(metrics.expensive_calls, metrics.cheap_calls)
        self.assertEqual([len(batch) for batch in cheap_batches], [4] * 5)
        self.assertEqual([len(batch) for batch in expensive_batches], [12, 8])

    def test_missing_cheap_decision_fails_open_for_that_candidate(self) -> None:
        movies = [
            {"movie_id": "tt1", "text": "movie 1"},
            {"movie_id": "tt2", "text": "movie 2"},
        ]
        reviews = [
            {"tconst": "tt1", "review": "one", "text": "one"},
            {"tconst": "tt2", "review": "two", "text": "two"},
        ]
        expensive_batches = []

        join = CascadeJoin(
            CascadeConfig(
                cheap_batch_size=2,
                expensive_batch_size=4,
                calibration_budget=1,
            ),
            cheap_score_batch=lambda candidates, predicate: {
                candidates[0].candidate_id: -2.0
            },
            expensive_classify=lambda candidates, predicate: (
                expensive_batches.append(candidates) or set()
            ),
        )
        _, decisions, metrics = join.run(movies, reviews, "negative")

        self.assertEqual([item.route for item in decisions], [
            "cheap_reject",
            "expensive",
        ])
        self.assertEqual(metrics.cheap_failure_candidates, 1)
        self.assertEqual(metrics.calibration_expensive_calls, 1)
        self.assertEqual(metrics.fallback_expensive_calls, 1)
        self.assertEqual(len(expensive_batches), 2)

    def test_time_percentages_use_model_call_time(self) -> None:
        movies = [{"movie_id": "tt1", "text": "movie"}]
        reviews = [{"tconst": "tt1", "review": "review", "text": "review"}]
        join = CascadeJoin(
            CascadeConfig(cheap_batch_size=1, expensive_batch_size=2),
            cheap_score_batch=lambda candidates, predicate: {
                candidates[0].candidate_id: -2.0
            },
            expensive_classify=lambda candidates, predicate: set(),
        )
        _, _, metrics = join.run(movies, reviews, "negative")
        self.assertAlmostEqual(
            metrics.cheap_time_percent + metrics.expensive_time_percent,
            100.0,
        )

    def test_expensive_batch_must_exceed_cheap_batch(self) -> None:
        with self.assertRaises(ValueError):
            CascadeJoin(
                CascadeConfig(cheap_batch_size=8, expensive_batch_size=8)
            )

    def test_parses_gemma_fenced_and_mapping_batch_json(self) -> None:
        payload = {
            "message": {
                "content": 'Result:\n```json\n{"answers":{"1":"yes","2":"no"}}\n```'
            }
        }
        self.assertEqual(
            _parse_batch_scores(payload, {1, 2}),
            {1: 2.0, 2: -2.0},
        )


if __name__ == "__main__":
    unittest.main()
