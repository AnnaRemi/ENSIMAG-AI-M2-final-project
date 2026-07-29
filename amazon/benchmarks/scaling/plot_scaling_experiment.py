#!/usr/bin/env python3
"""Plot cost/calls/time/quality vs candidate-pool size N, to find and mark
the crossover where a batched cascade (trummer_v1) becomes cheaper than the
vanilla per-row baseline (suql_baseline) -- backs the cascade cost-model
claim: baseline cost is O(N) with no fixed overhead; a cascade pays a fixed
calibration cost that only pays off once N is large enough to amortize it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent.parent.parent  # data_amazon/benchmarks/scaling -> repo root
sys.path.insert(0, str(LAB_ROOT / "generalized_pipeline"))
import pipeline as p  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_COLORS = {
    "suql_baseline": "#4c78a8",
    "suql_v1": "#f58518",
    "trummer_baseline": "#54a24b",
    "trummer_v1": "#6f5bd3",
}
METHOD_MARKERS = {
    "suql_baseline": "o",
    "suql_v1": "s",
    "trummer_baseline": "^",
    "trummer_v1": "D",
}


def find_crossover(frame: pd.DataFrame, baseline: str, cascade: str, value_col: str) -> float | None:
    """Smallest tested N at which cascade's value_col drops below baseline's.

    Cost/calls/time alone can be misleading: a cascade whose routing has
    collapsed (e.g. calibration_budget exceeding the whole candidate pool at
    small N) trivially looks "cheap" because it does little real work, not
    because it's efficient. Pair this with find_quality_gated_crossover.
    """
    base = frame[frame.method == baseline].sort_values("n")
    casc = frame[frame.method == cascade].sort_values("n")
    merged = pd.merge(base[["n", value_col]], casc[["n", value_col]], on="n", suffixes=("_base", "_casc"))
    below = merged[merged[f"{value_col}_casc"] < merged[f"{value_col}_base"]]
    return float(below["n"].min()) if not below.empty else None


def find_quality_gated_crossover(
    frame: pd.DataFrame, baseline: str, cascade: str, value_col: str,
    min_f1_ratio: float = 0.85,
) -> float | None:
    """Smallest tested N at which the cascade is both cheaper/faster AND its
    F1 is within min_f1_ratio of the baseline's -- the meaningful "cascading
    actually pays off here" point, as opposed to a degenerate-quality cheap
    result from a broken-small-N cascade.
    """
    base = frame[frame.method == baseline].sort_values("n")
    casc = frame[frame.method == cascade].sort_values("n")
    merged = pd.merge(
        base[["n", value_col, "f1"]], casc[["n", value_col, "f1"]], on="n", suffixes=("_base", "_casc"),
    )
    ok = merged[
        (merged[f"{value_col}_casc"] < merged[f"{value_col}_base"])
        & (merged["f1_casc"] >= min_f1_ratio * merged["f1_base"])
    ]
    return float(ok["n"].min()) if not ok.empty else None


def plot_metric_vs_n(
    frame: pd.DataFrame, value_col: str, ylabel: str, title: str, path: Path,
    baseline: str, cascade: str, log_y: bool = False, annotate: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for method, group in frame.groupby("method"):
        group = group.sort_values("n")
        ax.plot(
            group["n"], group[value_col], marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS.get(method), label=method, linewidth=2, markersize=8,
        )
    crossover = find_crossover(frame, baseline, cascade, value_col) if annotate else None
    gated = find_quality_gated_crossover(frame, baseline, cascade, value_col) if annotate else None
    if crossover is not None and crossover != gated:
        ax.axvline(crossover, color="#999999", linestyle=":", alpha=.6)
        ax.text(
            crossover, ax.get_ylim()[1] * .80, f"  {cascade} cheaper\n  from N={crossover:.0f}\n  (quality not gated)",
            fontsize=8, va="top", color="#666666",
        )
    if gated is not None:
        ax.axvline(gated, color="#333333", linestyle="--", alpha=.8)
        ax.text(
            gated, ax.get_ylim()[1] * .98, f"  {cascade} cheaper AND F1>={{:.0f}}% of {baseline}\n  from N={gated:.0f}".format(85),
            fontsize=9, va="top", fontweight="bold",
        )
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set(title=title, xlabel="Candidate pool size N (log scale)", ylabel=ylabel)
    ax.legend()
    ax.grid(alpha=.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="scaling_experiment.csv from run_scaling_experiment.py")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--accelerator-usd-per-hour", type=float, default=p.DEFAULT_ACCELERATOR_USD_PER_HOUR)
    ap.add_argument("--baseline", default="suql_baseline")
    ap.add_argument("--cascade", default="trummer_v1")
    args = ap.parse_args()

    raw = pd.read_csv(args.input)
    numeric = [
        "precision", "recall", "f1", "wall_seconds", "llm_calls",
        "cheap_calls", "expensive_calls", "cheap_seconds", "expensive_seconds",
    ]
    frame = raw.groupby(["n", "method"], as_index=False)[numeric].mean()
    frame = p.add_cost_columns(frame, args.accelerator_usd_per_hour)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_vs_n(
        frame, "total_cost_usd", "Estimated cost (USD; lower is better)",
        "Cost vs candidate pool size", args.output_dir / "01_cost_vs_n.png",
        args.baseline, args.cascade,
    )
    plot_metric_vs_n(
        frame, "llm_calls", "Mean LLM calls (lower is better)",
        "LLM calls vs candidate pool size", args.output_dir / "02_calls_vs_n.png",
        args.baseline, args.cascade,
    )
    plot_metric_vs_n(
        frame, "wall_seconds", "Wall time in seconds (lower is better)",
        "Wall time vs candidate pool size", args.output_dir / "03_time_vs_n.png",
        args.baseline, args.cascade,
    )
    plot_metric_vs_n(
        frame, "f1", "F1 (higher is better)",
        "Quality vs candidate pool size", args.output_dir / "04_f1_vs_n.png",
        args.baseline, args.cascade, annotate=False,
    )

    frame.to_csv(args.output_dir / "scaling_summary.csv", index=False)
    for value_col, label in (("total_cost_usd", "cost"), ("wall_seconds", "wall time"), ("llm_calls", "calls")):
        crossover = find_crossover(frame, args.baseline, args.cascade, value_col)
        gated = find_quality_gated_crossover(frame, args.baseline, args.cascade, value_col)
        if crossover is not None:
            print(f"{label}: {args.cascade} cheaper than {args.baseline} from N>={crossover:.0f} (quality not gated)")
        else:
            print(f"{label}: {args.cascade} never cheaper than {args.baseline} in the tested range")
        if gated is not None:
            print(f"{label}: {args.cascade} cheaper AND F1>=85% of {args.baseline} from N>={gated:.0f} (meaningful crossover)")
        else:
            print(f"{label}: no N tested where {args.cascade} was both cheaper and quality-competitive")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
