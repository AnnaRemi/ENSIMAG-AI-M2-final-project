#!/usr/bin/env python3
"""Run a SUQL engine implementation from one shared CLI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE_DIR = ROOT / "src_baseline"

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


def load_engine(engine_dir: Path):
    engine_dir = engine_dir.resolve()
    if not (engine_dir / "suql_engine.py").exists():
        raise FileNotFoundError(f"No suql_engine.py found in {engine_dir}")
    sys.path.insert(0, str(engine_dir))
    try:
        from suql_engine import ask, ask_with_suql
    finally:
        sys.path.pop(0)
    return ask, ask_with_suql


def run_all_examples(engine_dir: Path, output_dir: Path, verbose: bool = True) -> dict[str, pd.DataFrame]:
    ask, _ = load_engine(engine_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for example in EXAMPLE_QUESTIONS:
        print(f"\n{'=' * 70}")
        print(f"[{example['id']}] {example['description']}")
        out_path = output_dir / f"{example['id']}_results.csv"
        results[example["id"]] = ask(example["question"], output_csv=str(out_path), verbose=verbose)

    print(f"\n{'=' * 70}")
    print(f"All results saved to: {output_dir}/")
    return results


def run_single(engine_dir: Path, output_dir: Path, question: str, verbose: bool = True, output_csv: str | None = None) -> pd.DataFrame:
    ask, _ = load_engine(engine_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_csv or str(output_dir / "custom_question_results.csv")
    return ask(question, output_csv=out_path, verbose=verbose)


def run_suql(engine_dir: Path, output_dir: Path, suql_query: str, verbose: bool = True, output_csv: str | None = None) -> pd.DataFrame:
    _, ask_with_suql = load_engine(engine_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_csv or str(output_dir / "custom_suql_results.csv")
    return ask_with_suql(suql_query, output_csv=out_path, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SUQL movie engine implementation.")
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--output-dir", type=Path, help="Default output directory. Defaults to <engine-dir>/outputs.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("question", nargs="?", help="Natural-language question.")
    group.add_argument("--suql", metavar="QUERY", help="Raw SUQL query string to execute directly.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose progress output.")
    parser.add_argument("--output", metavar="CSV", help="Custom output CSV path for a single question or raw SUQL query.")
    args = parser.parse_args()

    engine_dir = args.engine_dir.resolve()
    output_dir = args.output_dir or (engine_dir / "outputs")
    verbose = not args.quiet

    if args.suql:
        run_suql(engine_dir, output_dir, args.suql, verbose=verbose, output_csv=args.output)
    elif args.question:
        run_single(engine_dir, output_dir, args.question, verbose=verbose, output_csv=args.output)
    else:
        run_all_examples(engine_dir, output_dir, verbose=verbose)


if __name__ == "__main__":
    main()
