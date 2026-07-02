from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trummer_join.data import load_movies_and_reviews
from trummer_join.structured_filter import (
    apply_structured_filters,
    extract_structured_filters,
    semantic_predicate_from_question,
)


class StructuredPruningTests(unittest.TestCase):
    def test_filters_year_and_reviews_before_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "imdb_structured_joined.csv").write_text(
                "\n".join(
                    [
                        "movie_id,title,year,director,runtime,genres",
                        "tt1,One,1998,A,90,Drama",
                        "tt2,Two,1999,B,100,Drama",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (data_dir / "imdb_reviews.csv").write_text(
                "\n".join(
                    [
                        "tconst,review",
                        "tt1,bad movie",
                        "tt2,bad but wrong year",
                        "tt3,unrelated",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            movies, reviews = load_movies_and_reviews(data_dir=data_dir, year=1998)

        self.assertEqual(movies["movie_id"].tolist(), ["tt1"])
        self.assertEqual(reviews["tconst"].tolist(), ["tt1"])

    def test_extracts_year_and_genre_from_question(self) -> None:
        frame = pd.DataFrame(
            [
                {"movie_id": "tt1", "year": 2001, "genres": "Drama,Romance"},
                {"movie_id": "tt2", "year": 2001, "genres": "Comedy"},
                {"movie_id": "tt3", "year": 1998, "genres": "Drama"},
            ]
        )

        filters = extract_structured_filters(
            "Find 2001 year drama movies with positive reviews",
            frame.columns,
        )
        filtered = apply_structured_filters(frame, filters)

        self.assertEqual([item.as_dict() for item in filters], [
            {"column": "year", "op": "eq", "value": "2001", "source": "2001"},
            {"column": "genres", "op": "contains", "value": "Drama", "source": "drama"},
        ])
        self.assertEqual(filtered["movie_id"].tolist(), ["tt1"])
        self.assertIn("positive", semantic_predicate_from_question("2001 drama movies with positive reviews"))


if __name__ == "__main__":
    unittest.main()
