from __future__ import annotations

import unittest

from trummer_join.operators import parse_pairs


class StructuredOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.movies = [
            {"movie_id": "tt1", "title": "one"},
            {"movie_id": "tt2", "title": "two"},
        ]
        self.reviews = [
            {"tconst": "tt2", "review": "bad"},
            {"tconst": "tt1", "review": "awful"},
        ]

    def test_parses_json_pairs(self) -> None:
        rows = parse_pairs(
            '{"pairs":[{"movie_id":"tt1","tconst":"tt1"}]}',
            self.movies,
            self.reviews,
        )
        self.assertEqual([(row["movie_id"], row["tconst"]) for row in rows], [("tt1", "tt1")])

    def test_parses_matching_movie_ids(self) -> None:
        rows = parse_pairs(
            '{"matching_movie_ids":["tt1"]}',
            self.movies,
            self.reviews,
        )
        self.assertEqual(
            [(row["movie_id"], row["tconst"]) for row in rows],
            [("tt1", "tt1")],
        )

    def test_rejects_structurally_invalid_pair(self) -> None:
        rows = parse_pairs(
            '{"pairs":[{"movie_id":"tt1","tconst":"tt2"}]}',
            self.movies,
            self.reviews,
        )
        self.assertEqual(rows, [])

    def test_keeps_legacy_pair_fallback(self) -> None:
        rows = parse_pairs("2,1", self.movies, self.reviews)
        self.assertEqual([(row["movie_id"], row["tconst"]) for row in rows], [("tt2", "tt2")])


if __name__ == "__main__":
    unittest.main()
