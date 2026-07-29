#!/usr/bin/env python3
"""Build paper-oriented analysis plots from existing benchmark outputs only
-- no new experiments, no LLM calls. Reads:
  - a 10q/Nrep comparison.csv (per question x method x repetition rows,
    including calibration_agreement/calibration_candidates)
  - the same run's per-(question, method, repetition) decisions.csv files
    (row-level "source" -- cheap/calibration_expensive/expensive/
    expensive_batch -- showing how each cascade actually routed)
  - a row-count scaling_experiment.csv (optional)

Produces figures not already covered by generalized_pipeline's plot_suite:
  01_calibration_agreement.png  -- how often the cascades' calibration
                                    actually met cascade_target=0.9
  02_routing_breakdown.png      -- what fraction of final decisions came
                                    from cheap vs expensive vs calibration
                                    per method, aggregated over the whole run
  03_f1_heatmap.png             -- per-question x per-method F1, showing
                                    which concepts each method struggles on
  04_precision_recall_scatter.png -- operating region of each method across
                                    all questions
  05_cost_savings_vs_n.png      -- % cost reduction of the cascade vs the
                                    vanilla baseline as N grows (needs
                                    --scaling-csv)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_COLORS = {
    "suql_baseline": "#4c78a8",
    "suql_v1": "#f58518",
    "trummer_baseline": "#54a24b",
    "trummer_v1": "#6f5bd3",
}
METHOD_ORDER = ["suql_baseline", "suql_v1", "trummer_baseline", "trummer_v1"]
SOURCE_COLORS = {
    "cheap": "#4c9f70",
    "calibration_expensive": "#f2c744",
    "expensive": "#d95f02",
    "expensive_batch": "#a63603",
}
SOURCE_LABELS = {
    "cheap": "Cheap model (trusted)",
    "calibration_expensive": "Expensive (calibration sample)",
    "expensive": "Expensive (escalated, per-row)",
    "expensive_batch": "Expensive (escalated, batched)",
}


def load_comparison(outputs_dir: Path) -> pd.DataFrame:
    return pd.read_csv(outputs_dir / "comparison.csv")


def load_decision_sources(outputs_dir: Path) -> pd.DataFrame:
    """Aggregate the "source" column across every decisions.csv under
    outputs_dir/q_*/method/rep_*/decisions.csv into one long frame.
    """
    rows = []
    for path in sorted(outputs_dir.glob("q_*/*/rep_*/decisions.csv")):
        method = path.parent.parent.name
        question = path.parent.parent.parent.name
        frame = pd.read_csv(path, dtype=str)
        counts = frame["source"].fillna("expensive").value_counts()
        for source, count in counts.items():
            rows.append({"question": question, "method": method, "source": source, "count": int(count)})
    if not rows:
        raise FileNotFoundError(f"No decisions.csv found under {outputs_dir}/q_*/*/rep_*/")
    return pd.DataFrame(rows)


def plot_calibration_agreement(comparison: pd.DataFrame, path: Path, cascade_target: float = 0.9) -> None:
    cascade_methods = [m for m in ("suql_v1", "trummer_v1") if m in comparison["method"].unique()]
    data = [
        comparison.loc[comparison.method == m, "calibration_agreement"].dropna().to_numpy()
        for m in cascade_methods
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, tick_labels=cascade_methods, patch_artist=True, showmeans=True)
    for patch, method in zip(bp["boxes"], cascade_methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "#888888"))
        patch.set_alpha(.6)
    ax.axhline(cascade_target, color="#333333", linestyle="--", alpha=.7)
    ax.text(0.55, cascade_target, f" cascade_target={cascade_target}", va="bottom", fontsize=9)
    ax.set(
        title="Calibration agreement with the expensive oracle\n(per question x repetition)",
        ylabel="Agreement between cheap-model decision and expensive oracle", ylim=(-.03, 1.05),
    )
    ax.grid(axis="y", alpha=.25)
    fig.text(
        .5, .01,
        "When agreement falls below cascade_target, the learned threshold is None and every\n"
        "candidate escalates to the expensive model -- the cascade buys no savings that round.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_routing_breakdown(sources: pd.DataFrame, path: Path) -> None:
    totals = sources.groupby(["method", "source"], as_index=False)["count"].sum()
    pivot = totals.pivot(index="method", columns="source", values="count").fillna(0)
    pivot = pivot.reindex([m for m in METHOD_ORDER if m in pivot.index])
    shares = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bottom = np.zeros(len(shares))
    x = np.arange(len(shares))
    for source in sorted(shares.columns, key=lambda s: list(SOURCE_COLORS).index(s) if s in SOURCE_COLORS else 99):
        values = shares[source].to_numpy(float)
        ax.bar(
            x, values, bottom=bottom, label=SOURCE_LABELS.get(source, source),
            color=SOURCE_COLORS.get(source, "#888888"),
        )
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v >= 4:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottom += values
    ax.set_xticks(x, shares.index, rotation=12, ha="right")
    ax.set(title="Where each method's final decisions actually came from\n(share of all candidates, whole run)", ylabel="Share of decisions (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_f1_heatmap(comparison: pd.DataFrame, path: Path) -> None:
    agg = comparison.groupby(["question", "method"], as_index=False)["f1"].mean()
    pivot = agg.pivot(index="question", columns="method", values="f1")
    pivot = pivot[[m for m in METHOD_ORDER if m in pivot.columns]]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(pivot.to_numpy(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iat[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                        color="black" if .35 < value < .75 else "white")
    fig.colorbar(im, ax=ax, label="Mean F1")
    ax.set_title("F1 by question and method\n(mean over repetitions)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_precision_recall_scatter(comparison: pd.DataFrame, path: Path) -> None:
    agg = comparison.groupby(["question", "method"], as_index=False)[["precision", "recall"]].mean()
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for method, group in agg.groupby("method"):
        ax.scatter(
            group["recall"], group["precision"], label=method, s=90, alpha=.8,
            color=METHOD_COLORS.get(method, "#888888"), edgecolor="white", linewidth=.5,
        )
    ax.set(
        title="Precision vs recall across all questions\n(one point per question x method, mean over repetitions)",
        xlabel="Recall (higher is better)", ylabel="Precision (higher is better)", xlim=(-.03, 1.05), ylim=(-.03, 1.05),
    )
    ax.plot([0, 1], [0, 1], color="#cccccc", linestyle=":", zorder=0)
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cost_savings_vs_n(scaling: pd.DataFrame, path: Path, baseline: str, cascade: str) -> None:
    numeric = ["cheap_seconds", "expensive_seconds"]
    agg = scaling.groupby(["n", "method"], as_index=False)[numeric].mean()
    usd_per_second = 3.0 / 3600
    agg["total_cost_usd"] = (agg["cheap_seconds"] + agg["expensive_seconds"]) * usd_per_second
    base = agg[agg.method == baseline].set_index("n")["total_cost_usd"]
    casc = agg[agg.method == cascade].set_index("n")["total_cost_usd"]
    ns = sorted(set(base.index) & set(casc.index))
    savings = [100 * (1 - casc[n] / base[n]) for n in ns]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d95f02" if s < 0 else "#4c9f70" for s in savings]
    bars = ax.bar([str(n) for n in ns], savings, color=colors)
    for bar, s in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width() / 2, s, f"{s:.0f}%", ha="center",
                va="bottom" if s >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set(
        title=f"{cascade} cost savings vs {baseline} by candidate pool size N",
        xlabel="Candidate pool size N", ylabel="Cost reduction (%; positive = cascade cheaper)",
    )
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", type=Path, required=True, help="e.g. data_amazon/outputs/amazon_fashion_10q_10rep_.../")
    ap.add_argument("--scaling-csv", type=Path, default=None, help="optional scaling_experiment.csv")
    ap.add_argument("--plots-dir", type=Path, default=None)
    ap.add_argument("--baseline", default="suql_baseline")
    ap.add_argument("--cascade", default="trummer_v1")
    args = ap.parse_args()

    plots_dir = args.plots_dir or (args.outputs_dir / "paper_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    comparison = load_comparison(args.outputs_dir)
    sources = load_decision_sources(args.outputs_dir)

    plot_calibration_agreement(comparison, plots_dir / "01_calibration_agreement.png")
    plot_routing_breakdown(sources, plots_dir / "02_routing_breakdown.png")
    plot_f1_heatmap(comparison, plots_dir / "03_f1_heatmap.png")
    plot_precision_recall_scatter(comparison, plots_dir / "04_precision_recall_scatter.png")

    if args.scaling_csv is not None:
        scaling = pd.read_csv(args.scaling_csv)
        plot_cost_savings_vs_n(scaling, plots_dir / "05_cost_savings_vs_n.png", args.baseline, args.cascade)

    print(f"Wrote plots to {plots_dir}")
    for path in sorted(plots_dir.glob("*.png")):
        print(" -", path.name)


if __name__ == "__main__":
    main()
