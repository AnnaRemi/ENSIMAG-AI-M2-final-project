from __future__ import annotations

import unittest

from run_use_case3 import prune_inputs
from trummer_join.cascade import exact_id_candidates
from trummer_join.structured_filter import semantic_predicate_from_question


class StructuredCascadeTests(unittest.TestCase):
    def test_prunes_structured_columns_before_cascade_candidates(self) -> None:
        movies = [
            {"movie_id": "tt1", "year": "2001", "genres": "Drama", "text": "movie 1"},
            {"movie_id": "tt2", "year": "2001", "genres": "Comedy", "text": "movie 2"},
            {"movie_id": "tt3", "year": "1998", "genres": "Drama", "text": "movie 3"},
        ]
        reviews = [
            {"tconst": "tt1", "review": "great", "text": "great"},
            {"tconst": "tt2", "review": "great", "text": "great"},
            {"tconst": "tt3", "review": "great", "text": "great"},
        ]

        pruned_movies, pruned_reviews, filters = prune_inputs(
            movies,
            reviews,
            "Find 2001 year drama movies with positive reviews",
            fallback_year=None,
        )
        candidates = exact_id_candidates(pruned_movies, pruned_reviews)

        self.assertEqual([item.as_dict() for item in filters], [
            {"column": "year", "op": "eq", "value": "2001", "source": "2001"},
            {"column": "genres", "op": "contains", "value": "Drama", "source": "drama"},
        ])
        self.assertEqual([movie["movie_id"] for movie in pruned_movies], ["tt1"])
        self.assertEqual([review["tconst"] for review in pruned_reviews], ["tt1"])
        self.assertEqual([(item.movie["movie_id"], item.review["tconst"]) for item in candidates], [("tt1", "tt1")])
        self.assertIn("positive", semantic_predicate_from_question("2001 drama movies with positive reviews"))


if __name__ == "__main__":
    unittest.main()
