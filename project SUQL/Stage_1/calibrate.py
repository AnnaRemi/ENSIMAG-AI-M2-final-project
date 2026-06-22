#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import pandas as pd

from profiler import OllamaLogOddsScorer, profile_dataframe, save_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SUQL answer() log-odds thresholds.")
    parser.add_argument("input_csv", help="CSV with columns: review,question,label")
    parser.add_argument("--output", default="thresholds.json")
    parser.add_argument("--recall-target", type=float, default=0.9)
    parser.add_argument("--precision-target", type=float, default=0.7)
    parser.add_argument("--credible-level", type=float, default=0.95)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--model", default=os.environ.get("SUQL_MODEL", "ollama/phi4-mini"))
    parser.add_argument("--api-base", default=os.environ.get("SUQL_API_BASE", "http://localhost:11434"))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    scorer = OllamaLogOddsScorer(model=args.model, api_base=args.api_base)
    profile = profile_dataframe(
        df,
        scorer=scorer,
        recall_target=args.recall_target,
        precision_target=args.precision_target,
        credible_level=args.credible_level,
        grid_size=args.grid_size,
    )
    save_thresholds(profile, args.output)
    print(f"Wrote thresholds to {args.output}")


if __name__ == "__main__":
    main()
