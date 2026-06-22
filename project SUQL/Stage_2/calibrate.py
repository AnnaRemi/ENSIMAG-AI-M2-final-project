#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from profiler import DEFAULT_API_BASE, DEFAULT_CHEAP_MODEL, OllamaLogOddsScorer, profile_dataframe, save_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Stage_2 cheap-model cascade thresholds.")
    parser.add_argument("csv", help="CSV with columns: review,question,label")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "thresholds.json"))
    parser.add_argument("--cheap-model", default=os.environ.get("SUQL_CHEAP_MODEL", DEFAULT_CHEAP_MODEL))
    parser.add_argument("--api-base", default=os.environ.get("SUQL_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--accept-precision", type=float, default=0.9)
    parser.add_argument("--reject-precision", type=float, default=0.9)
    parser.add_argument("--credible-level", type=float, default=0.95)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    scorer = OllamaLogOddsScorer(model=args.cheap_model, api_base=args.api_base)
    profile = profile_dataframe(
        df,
        scorer=scorer,
        accept_precision_target=args.accept_precision,
        reject_precision_target=args.reject_precision,
        credible_level=args.credible_level,
    )
    save_thresholds(profile, args.output)
    print(f"Wrote Stage_2 thresholds to {args.output}")


if __name__ == "__main__":
    main()
