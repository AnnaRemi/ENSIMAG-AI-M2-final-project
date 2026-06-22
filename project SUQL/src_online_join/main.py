#!/usr/bin/env python3
"""
main.py
=======
Run a set of example questions against the online-join SUQL IMDb engine and
save results to CSV files in the outputs/ directory.

Usage:
    python main.py                      # run all example questions
    python main.py "your question here" # run a single custom question
    python main.py --suql "SELECT ..."  # run a raw SUQL query

Online-join execution:
  • Structured predicates run on ../data/imdb_structured_joined.csv.
  • answer(review, question) retrieval runs in parallel on ../data/imdb_reviews.csv.
  • The two retrieval outputs are joined on movie_id/tconst before final output.
  • summary(review) generates a short prose summary after the join.
"""

import sys
import os
import argparse
import pandas as pd

# Allow running from the project root
sys.path.insert(0, os.path.dirname(__file__))
from suql_engine import ask, ask_with_suql

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Example questions  (mirrors the kinds of queries in the SUQL paper)
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    {
        "id": "q1",
        "question": "What are the top 5 movies released in 1999 considered amazing in reviews?",
        "description": "Hybrid: structured year filter + LLM review sentiment",
    },
    {
        "id": "q2",
        "question": "Find drama movies from the 1990s that reviewers consider a masterpiece",
        "description": "Hybrid: genre + decade filter + LLM masterpiece detection",
    },
    {
        "id": "q3",
        "question": "Which horror movies under 100 minutes have terrifying reviews?",
        "description": "Hybrid: genre + runtime filter + LLM terror detection",
    },
    {
        "id": "q4",
        "question": "List 5 comedy movies after 2000 that reviewers found surprisingly funny",
        "description": "Hybrid: genre + year filter + LLM humor sentiment",
    },
    {
        "id": "q5",
        "question": "What are movies directed by Christopher Nolan with reviews praising the plot?",
        "description": "Hybrid: director name filter + LLM plot quality detection",
    },
]


def run_all_examples(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Run all example questions and save CSVs. Returns a dict of results."""
    results: dict[str, pd.DataFrame] = {}

    for ex in EXAMPLE_QUESTIONS:
        print(f"\n{'='*70}")
        print(f"[{ex['id']}] {ex['description']}")
        out_path = os.path.join(OUTPUT_DIR, f"{ex['id']}_results.csv")
        df = ask(ex["question"], output_csv=out_path, verbose=verbose)
        results[ex["id"]] = df

    print(f"\n{'='*70}")
    print(f"All results saved to: {OUTPUT_DIR}/")
    return results


def run_single(
    question: str,
    verbose: bool = True,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Run a single natural-language question."""
    out_path = output_csv or os.path.join(OUTPUT_DIR, "custom_question_results.csv")
    return ask(question, output_csv=out_path, verbose=verbose)


def run_suql(
    suql_query: str,
    verbose: bool = True,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Run a raw SUQL query."""
    out_path = output_csv or os.path.join(OUTPUT_DIR, "custom_suql_results.csv")
    return ask_with_suql(suql_query, output_csv=out_path, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SUQL Movie Database — query IMDb with natural language"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "question",
        nargs="?",
        help="Natural language question (e.g. 'top 5 amazing movies from 1999')",
    )
    group.add_argument(
        "--suql",
        metavar="QUERY",
        help="Raw SUQL query string to execute directly",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress output",
    )
    parser.add_argument(
        "--output",
        metavar="CSV",
        help="Custom output CSV path for a single question or raw SUQL query",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    if args.suql:
        run_suql(args.suql, verbose=verbose, output_csv=args.output)
    elif args.question:
        run_single(args.question, verbose=verbose, output_csv=args.output)
    else:
        run_all_examples(verbose=verbose)


if __name__ == "__main__":
    main()
