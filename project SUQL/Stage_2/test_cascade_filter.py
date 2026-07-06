from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade_filter import CascadeAnswerFilter


class FakeScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def score(self, review: str, question: str) -> float:
        del question
        return self.scores[review]


class FakeCascadeFilter(CascadeAnswerFilter):
    def __init__(self, scores: dict[str, float], expensive_answers: list[str], **kwargs) -> None:
        super().__init__(
            thresholds_path=None,
            cheap_model="ollama/cheap",
            expensive_model="ollama/expensive",
            **kwargs,
        )
        self.cheap_scorer = FakeScorer(scores)
        self.expensive_answers = list(expensive_answers)

    def expensive_answer(self, review_text: str, question: str) -> str:
        del review_text, question
        self.stats.expensive_full_calls += 1
        return self.expensive_answers.pop(0)


class CascadeAnswerFilterTests(unittest.TestCase):
    def test_learned_confidence_threshold_routes_by_sign(self) -> None:
        cascade = FakeCascadeFilter(
            scores={
                "accept review": 3.0,
                "reject review": -3.0,
                "uncertain review": 0.0,
            },
            expensive_answers=[
                "Yes",
                "No",
                "No",
                "Yes",
            ],
            cascade_target=1.0,
            calibration_budget=3,
        )

        results = cascade.answer_batch(
            ["accept review", "reject review", "uncertain review"],
            "Is this a match?",
        )

        self.assertEqual(results, ["Yes", "No", "Yes"])
        self.assertEqual(cascade.stats.cheap_early_accept, 1)
        self.assertEqual(cascade.stats.cheap_early_reject, 1)
        self.assertEqual(cascade.stats.calibration_expensive_calls, 3)
        self.assertEqual(cascade.stats.expensive_full_calls, 4)
        usage = cascade.stats.model_usage_by_question["Is this a match?"]
        self.assertEqual(usage["learned_confidence_threshold"], 3.0)
        self.assertEqual(usage["routing_confidence_threshold"], 3.0)

    def test_manual_confidence_threshold_skips_calibration(self) -> None:
        cascade = FakeCascadeFilter(
            scores={
                "accept review": 2.0,
                "reject review": -2.0,
                "uncertain review": 0.0,
            },
            expensive_answers=["Yes"],
            manual_confidence_threshold=2.0,
            calibration_budget=20,
        )

        results = cascade.answer_batch(
            ["accept review", "reject review", "uncertain review"],
            "Is this a match?",
        )

        self.assertEqual(results, ["Yes", "No", "Yes"])
        self.assertEqual(cascade.stats.calibration_expensive_calls, 0)
        self.assertEqual(cascade.stats.expensive_full_calls, 1)
        usage = cascade.stats.model_usage_by_question["Is this a match?"]
        self.assertIsNone(usage["learned_confidence_threshold"])
        self.assertEqual(usage["routing_confidence_threshold"], 2.0)


if __name__ == "__main__":
    unittest.main()
