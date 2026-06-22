from __future__ import annotations

import unittest

from trummer_join.cascade import (
    CascadeConfig,
    CascadeJoin,
    OllamaExpensiveClassifier,
    exact_id_candidates,
    extract_binary_log_odds,
)


class CascadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.movies = [
            {"movie_id": "tt1", "text": "movie 1"},
            {"movie_id": "tt2", "text": "movie 2"},
            {"movie_id": "tt3", "text": "movie 3"},
        ]
        self.reviews = [
            {"tconst": "tt1", "review": "accept", "text": "accept"},
            {"tconst": "tt2", "review": "reject", "text": "reject"},
            {"tconst": "tt3", "review": "uncertain", "text": "uncertain"},
            {"tconst": "tt9", "review": "unrelated", "text": "unrelated"},
        ]

    def test_candidates_are_structurally_keyed(self) -> None:
        candidates = exact_id_candidates(self.movies, self.reviews)
        self.assertEqual([item.candidate_id for item in candidates], [1, 2, 3])
        self.assertEqual(
            [(item.movie["movie_id"], item.review["tconst"]) for item in candidates],
            [("tt1", "tt1"), ("tt2", "tt2"), ("tt3", "tt3")],
        )

    def test_cascade_routes_and_batches_only_uncertain_candidates(self) -> None:
        seen_batches = []

        def score(candidate, predicate):
            del predicate
            return {"tt1": 3.0, "tt2": -3.0, "tt3": 0.0}[candidate.movie["movie_id"]]

        def classify(candidates, predicate):
            del predicate
            seen_batches.append([item.candidate_id for item in candidates])
            return {item.candidate_id for item in candidates}

        join = CascadeJoin(
            CascadeConfig(accept_threshold=1.5, reject_threshold=-1.5),
            cheap_score=score,
            expensive_classify=classify,
        )
        rows, decisions, metrics = join.run(self.movies, self.reviews, "negative")
        self.assertEqual([item.route for item in decisions], ["cheap_accept", "cheap_reject", "expensive"])
        self.assertEqual(seen_batches, [[3]])
        self.assertEqual([row["match_source"] for row in rows], ["cheap_accept", "expensive_accept"])
        self.assertEqual(metrics.expensive_calls, 1)

    def test_cheap_failure_falls_back_to_expensive(self) -> None:
        def fail(candidate, predicate):
            del candidate, predicate
            raise RuntimeError("offline")

        join = CascadeJoin(
            CascadeConfig(),
            cheap_score=fail,
            expensive_classify=lambda candidates, predicate: {item.candidate_id for item in candidates},
        )
        rows, _, metrics = join.run(self.movies[:1], self.reviews[:1], "negative")
        self.assertEqual(len(rows), 1)
        self.assertEqual(metrics.cheap_failures, 1)
        self.assertEqual(metrics.expensive_candidates, 1)

    def test_binary_log_odds_and_hard_response(self) -> None:
        payload = {"choices": [{"logprobs": {"top_logprobs": [{" 1": -0.2, " 0": -2.2}]}}]}
        self.assertAlmostEqual(extract_binary_log_odds(payload), 2.0)
        self.assertEqual(extract_binary_log_odds({"response": "0"}), -2.0)
        self.assertEqual(
            extract_binary_log_odds({"message": {"content": '{"answer": 1}'}}),
            2.0,
        )

    def test_threshold_validation(self) -> None:
        with self.assertRaises(ValueError):
            CascadeJoin(CascadeConfig(accept_threshold=0, reject_threshold=0))

    def test_expensive_parser_accepts_pair_labels_and_unique_movie_ids(self) -> None:
        classifier = OllamaExpensiveClassifier(CascadeConfig())
        candidates = exact_id_candidates(self.movies, self.reviews)
        answer = "PAIR_1, tt3"
        # Exercise the response parser without a network request.
        import trummer_join.cascade as module

        original = module._post_json
        module._post_json = lambda *args, **kwargs: {"message": {"content": answer}}
        try:
            self.assertEqual(classifier.classify(candidates, "negative"), {1, 3})
        finally:
            module._post_json = original


if __name__ == "__main__":
    unittest.main()
