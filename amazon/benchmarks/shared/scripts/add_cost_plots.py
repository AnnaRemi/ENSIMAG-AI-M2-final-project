#!/usr/bin/env python3
"""Add cost plots to every existing benchmark plots directory."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate_and_plot import (
    DEFAULT_ACCELERATOR_USD_PER_HOUR,
    add_cost_columns,
    plot_cost_f1,
    plot_costs,
)


def frame_for(plots_dir: Path) -> tuple[pd.DataFrame, Path, str] | None:
    owner = plots_dir.parent
    if owner.name.startswith("q_"):
        output = owner.parent.parent
        comparison = output / "comparison.csv"
        if not comparison.exists():
            return None
        frame = pd.read_csv(comparison)
        frame = frame[frame["question"].astype(str) == owner.name].copy()
        return frame, comparison, owner.name
    aggregate = owner / "aggregate.csv"
    if aggregate.exists():
        suite = owner.parent.parent.name if owner.parent.name == "outputs" else owner.name
        return pd.read_csv(aggregate), aggregate, f"{suite} averaged experiment"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--accelerator-usd-per-hour", type=float, default=DEFAULT_ACCELERATOR_USD_PER_HOUR)
    args = parser.parse_args()
    made = 0
    for plots_dir in sorted(args.benchmarks_root.rglob("plots")):
        loaded = frame_for(plots_dir)
        if loaded is None:
            continue
        frame, csv_path, title = loaded
        if frame.empty:
            continue
        costed = add_cost_columns(frame, args.accelerator_usd_per_hour)
        # Persist the assumptions and derived values alongside the plots.
        if csv_path.name == "aggregate.csv":
            costed.to_csv(csv_path, index=False)
        plot_costs(costed, plots_dir / "05_cost_by_approach.png", title, args.accelerator_usd_per_hour)
        plot_cost_f1(costed, plots_dir / "06_cost_vs_f1.png", title, args.accelerator_usd_per_hour)
        made += 1
    print(f"Added two cost plots to {made} plot directories.")


if __name__ == "__main__":
    main()
