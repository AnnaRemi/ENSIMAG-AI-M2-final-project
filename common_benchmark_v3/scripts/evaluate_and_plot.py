#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    args = parser.parse_args()
    outputs_dir = Path(args.outputs_dir)
    truth = set(json.loads((ROOT / "benchmark.json").read_text())["ground_truth_movie_ids"])
    rows = []
    outcomes = []
    for path in sorted(outputs_dir.glob("*/run_metrics.json")):
        run = json.loads(path.read_text())
        found = set(run["found_movie_ids"])
        tp, fp, fn = len(found & truth), len(found - truth), len(truth - found)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "implementation": run["implementation"],
            "mode": run["mode"],
            "model": run["model"],
            "cheap_model": run.get("cheap_model", ""),
            "expensive_model": run.get("expensive_model", ""),
            "wall_seconds": run["wall_seconds"],
            "llm_calls": run["llm_calls"],
            "cheap_calls": run.get("cheap_calls", 0),
            "expensive_calls": run.get("expensive_calls", 0),
            "cheap_early_accepts": run.get("cheap_early_accepts", 0),
            "cheap_early_rejects": run.get("cheap_early_rejects", 0),
            "expensive_candidates": run.get("expensive_candidates", 0),
            "final_answer_rows": run["final_answer_rows"],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        for movie_id in sorted(truth | found):
            outcomes.append({
                "implementation": run["implementation"],
                "movie_id": movie_id,
                "ground_truth": int(movie_id in truth),
                "found": int(movie_id in found),
                "classification": "TP" if movie_id in truth and movie_id in found else "FP" if movie_id in found else "FN",
            })
    if len(rows) != 2:
        raise SystemExit(f"Expected two run_metrics.json files under {outputs_dir}, found {len(rows)}")
    frame = pd.DataFrame(rows).sort_values("implementation")
    frame.to_csv(outputs_dir / "comparison.csv", index=False)
    pd.DataFrame(outcomes).to_csv(outputs_dir / "movie_id_outcomes.csv", index=False)
    write_markdown(frame, outputs_dir / "comparison.md")
    plot(frame, outputs_dir / "comparison.png")
    print(frame.to_string(index=False))


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "implementation", "wall_seconds", "llm_calls", "cheap_calls",
        "cheap_early_accepts", "cheap_early_rejects", "expensive_calls",
        "precision", "recall", "f1",
    ]
    lines = [
        "# Trummer heterogen_v1 vs heterogen_v2 cascade",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    v1 = frame.set_index("implementation").loc["trummer_heterogen_v1"]
    v2 = frame.set_index("implementation").loc["trummer_heterogen_v2_cascade"]
    lines += [
        "",
        "Both implementations receive all 50 movies and all 50 reviews under the same semantic predicate.",
        "For v2, `llm_calls = cheap_calls + expensive_calls`; v1 uses block-join calls only.",
        "",
        "## Routing interpretation",
        "",
        f"- V1 issued {int(v1.llm_calls)} expensive block-join calls.",
        f"- V2 issued {int(v2.cheap_calls)} cheap calls and {int(v2.expensive_calls)} expensive calls.",
        f"- V2 early-accepted {int(v2.cheap_early_accepts)} candidates and early-rejected "
        f"{int(v2.cheap_early_rejects)} candidates.",
        f"- {int(v2.expensive_candidates)} candidates entered the uncertainty band.",
        "- If expensive calls are zero, the run measures cheap-model routing rather than a meaningful "
        "cheap-to-expensive fallback. Threshold calibration is required before treating that configuration "
        "as an effective cascade.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot(frame: pd.DataFrame, path: Path) -> None:
    labels = ["heterogen_v1", "heterogen_v2"]
    indexed = frame.set_index("implementation")
    ordered = indexed.loc[["trummer_heterogen_v1", "trummer_heterogen_v2_cascade"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(labels, ordered["wall_seconds"])
    axes[0].set_title("Wall time")
    axes[0].set_ylabel("Seconds")
    axes[1].bar(labels, ordered["cheap_calls"], label="cheap")
    axes[1].bar(labels, ordered["expensive_calls"], bottom=ordered["cheap_calls"], label="expensive")
    axes[1].set_title("LLM calls by stage")
    axes[1].legend()
    ordered[["precision", "recall", "f1"]].set_axis(labels).plot(kind="bar", ax=axes[2], rot=0)
    axes[2].set_title("Retrieval quality")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
