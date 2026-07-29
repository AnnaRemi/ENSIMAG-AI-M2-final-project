#!/usr/bin/env python3
"""Test Prop 2 (the cascade's cost advantage gets more *reliable*, not just
better on average, as N grows past the crossover n*): merges a many-
repetition run at one or two fixed N back into the master scaling-experiment
CSV, then measures how often a single repetition's realized cascade cost
exceeded vanilla's -- and whether that fraction shrinks as N grows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent.parent.parent  # data_amazon/benchmarks/scaling -> repo root
sys.path.insert(0, str(LAB_ROOT / "generalized_pipeline"))
import pipeline as p  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def merge_into_master(master_csv: Path, new_csv: Path) -> pd.DataFrame:
    """Replace whatever (n, method) rows the new run covers with its (fuller)
    data, keeping every other row from the master untouched. Writes back to
    master_csv in place ("load results to the same scaling experiment").
    """
    master = pd.read_csv(master_csv)
    new = pd.read_csv(new_csv)
    covered = set(zip(new["n"], new["method"]))
    kept = master[~master.apply(lambda r: (r["n"], r["method"]) in covered, axis=1)]
    merged = pd.concat([kept, new], ignore_index=True)
    merged = merged.sort_values(["n", "method", "repetition"]).reset_index(drop=True)
    merged.to_csv(master_csv, index=False)
    return merged


def realized_cost(frame: pd.DataFrame, accelerator_usd_per_hour: float) -> pd.DataFrame:
    return p.add_cost_columns(frame, accelerator_usd_per_hour)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion -- appropriate
    here because the failure rate can be small (near 0 or 1), where a naive
    normal approximation (p +- z*sqrt(p(1-p)/n)) can go negative or exceed 1.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half_width = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def reliability_table(
    frame: pd.DataFrame, ns: list[int], baseline: str, cascade: str, accelerator_usd_per_hour: float,
) -> pd.DataFrame:
    costed = realized_cost(frame, accelerator_usd_per_hour)
    rows = []
    for n in ns:
        base = costed[(costed.n == n) & (costed.method == baseline)].sort_values("repetition")
        casc = costed[(costed.n == n) & (costed.method == cascade)].sort_values("repetition")
        merged = pd.merge(
            base[["repetition", "total_cost_usd"]], casc[["repetition", "total_cost_usd"]],
            on="repetition", suffixes=("_base", "_casc"),
        )
        if merged.empty:
            continue
        merged["cascade_lost"] = merged["total_cost_usd_casc"] > merged["total_cost_usd_base"]
        merged["cost_ratio"] = merged["total_cost_usd_casc"] / merged["total_cost_usd_base"]
        failures = int(merged["cascade_lost"].sum())
        reps = len(merged)
        ci_low, ci_high = wilson_interval(failures, reps)
        rows.append({
            "n": n, "repetitions": reps, "failures": failures,
            "fraction_cascade_more_expensive": merged["cascade_lost"].mean(),
            "ci_low": ci_low, "ci_high": ci_high,
            "mean_cost_ratio": merged["cost_ratio"].mean(),
            "std_cost_ratio": merged["cost_ratio"].std(),
            "min_cost_ratio": merged["cost_ratio"].min(),
            "max_cost_ratio": merged["cost_ratio"].max(),
        })
    return pd.DataFrame(rows)


def plot_reliability(table: pd.DataFrame, path: Path, baseline: str, cascade: str) -> None:
    table = table.sort_values("n")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = table["fraction_cascade_more_expensive"].to_numpy(float) * 100
    lower = (table["fraction_cascade_more_expensive"] - table["ci_low"]).to_numpy(float) * 100
    upper = (table["ci_high"] - table["fraction_cascade_more_expensive"]).to_numpy(float) * 100
    ax.errorbar(
        table["n"], y, yerr=[lower, upper], marker="o", markersize=8, capsize=5,
        color="#d95f02", linewidth=2, ecolor="#d95f02", elinewidth=1.5,
    )
    for n, frac, failures, reps in zip(table["n"], y, table["failures"], table["repetitions"]):
        ax.annotate(f"{failures}/{reps}", (n, frac), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
    ax.set(
        title=f"Failure rate of {cascade} vs {baseline}, with 95% Wilson CI\n(fraction of single runs where the cascade's realized cost exceeded vanilla's)",
        xlabel="Candidate pool size N", ylabel="Share of repetitions where cascade cost > vanilla cost (%)",
        ylim=(-2, max(15, float(table["ci_high"].max() * 105 + 2))),
    )
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cost_ratio_distribution(frame: pd.DataFrame, ns: list[int], path: Path, baseline: str, cascade: str, accelerator_usd_per_hour: float) -> None:
    costed = realized_cost(frame, accelerator_usd_per_hour)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    data, labels = [], []
    for n in ns:
        base = costed[(costed.n == n) & (costed.method == baseline)].sort_values("repetition")
        casc = costed[(costed.n == n) & (costed.method == cascade)].sort_values("repetition")
        merged = pd.merge(
            base[["repetition", "total_cost_usd"]], casc[["repetition", "total_cost_usd"]],
            on="repetition", suffixes=("_base", "_casc"),
        )
        if merged.empty:
            continue
        ratio = merged["total_cost_usd_casc"] / merged["total_cost_usd_base"]
        data.append(ratio.to_numpy())
        labels.append(f"N={n}\n(n={len(ratio)})")
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#6f5bd3")
        patch.set_alpha(.6)
    ax.axhline(1.0, color="#333333", linestyle="--", alpha=.7)
    ax.text(len(data) + .45, 1.0, " cost parity", va="center", fontsize=9)
    ax.set(
        title=f"Per-repetition cost ratio: {cascade} / {baseline}\n(below 1.0 = cascade cheaper that run)",
        ylabel="Cost ratio (cascade / vanilla)",
    )
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master-csv", type=Path, required=True, help="the scaling experiment CSV to update in place")
    ap.add_argument("--new-csv", type=Path, required=True, help="the fresh many-repetition run's output CSV")
    ap.add_argument("--ns", type=int, nargs="+", required=True, help="the N values that were re-run with many repetitions")
    ap.add_argument("--baseline", default="suql_baseline")
    ap.add_argument("--cascade", default="trummer_v1")
    ap.add_argument("--accelerator-usd-per-hour", type=float, default=p.DEFAULT_ACCELERATOR_USD_PER_HOUR)
    ap.add_argument("--plots-dir", type=Path, default=None)
    args = ap.parse_args()

    merged = merge_into_master(args.master_csv, args.new_csv)
    print(f"Merged {args.new_csv} into {args.master_csv} ({len(merged)} total rows)")

    table = reliability_table(merged, args.ns, args.baseline, args.cascade, args.accelerator_usd_per_hour)
    print(table.to_string(index=False))

    plots_dir = args.plots_dir or (args.master_csv.parent / "reliability_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability(table, plots_dir / "01_cascade_loss_rate.png", args.baseline, args.cascade)
    plot_cost_ratio_distribution(merged, args.ns, plots_dir / "02_cost_ratio_distribution.png", args.baseline, args.cascade, args.accelerator_usd_per_hour)
    table.to_csv(plots_dir / "reliability_table.csv", index=False)
    print(f"\nWrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
