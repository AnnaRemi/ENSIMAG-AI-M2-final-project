#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from common import ROOT
from evaluate_and_plot import (
    plot_question_call_split,
    plot_question_quality,
    plot_question_time_split,
    plot_question_tradeoff,
)


DEFAULT_DIR = ROOT / "heterogen_struc_cascade"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate aggregate Heterogen structural/cascade plots."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    aggregate_path = output_dir / "aggregate_excluding_suql_baseline.csv"
    aggregate = pd.read_csv(aggregate_path)
    note = (
        "Heterogen-only aggregate: SUQL baseline is excluded. Each value is the "
        "mean across the 10 benchmark questions after each question/method was "
        "averaged over 11 repetitions."
    )

    plot_question_quality(
        aggregate,
        output_dir / "01_quality_precision_recall_f1.png",
        "Heterogen structural/cascade aggregate",
        note,
    )
    plot_question_time_split(
        aggregate,
        output_dir / "02_time_cheap_expensive_percent.png",
        "Heterogen structural/cascade aggregate",
        note,
    )
    plot_question_call_split(
        aggregate,
        output_dir / "03_calls_cheap_expensive_percent.png",
        "Heterogen structural/cascade aggregate",
        note,
    )
    plot_question_tradeoff(
        aggregate,
        output_dir / "04_quality_time_calls_tradeoff.png",
        "Heterogen structural/cascade aggregate",
        note,
    )
    print(f"Regenerated plots in {output_dir}")


if __name__ == "__main__":
    main()
