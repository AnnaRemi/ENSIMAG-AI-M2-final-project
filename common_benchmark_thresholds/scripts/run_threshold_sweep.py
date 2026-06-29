#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parent
V3_ROOT = LAB_ROOT / "common_benchmark_v3"
V3_SCRIPTS = V3_ROOT / "scripts"
sys.path.insert(0, str(V3_SCRIPTS))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from common import pair_slug
from repetitions import run_repeated


IMPLEMENTATIONS = {
    "v2": {
        "label": "Row-wise cascade",
        "script": "run_heterogen_v2.py",
        "extra": lambda args: [
            "--expensive-batch-size",
            str(args.expensive_batch_size),
        ],
    },
    "v2_3": {
        "label": "Batch-wise cascade",
        "script": "run_heterogen_v2_3.py",
        "extra": lambda args: [
            "--cheap-batch-size",
            str(args.cheap_batch_size),
            "--expensive-batch-size",
            str(args.v2_3_expensive_batch_size),
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep manual cascade thresholds for common benchmark v3."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="ollama/gemma3:270m")
    parser.add_argument("--expensive-model", default="ollama/gemma3:1b")
    parser.add_argument(
        "--thresholds",
        default="0,0.5,1,1.5,2,2.5,3",
        help="Comma-separated manual confidence thresholds to sweep.",
    )
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--calibration-budget", type=int, default=20)
    parser.add_argument("--cascade-target", type=float, default=0.9)
    parser.add_argument("--cheap-batch-size", type=int, default=8)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--v2-3-expensive-batch-size", type=int, default=32)
    parser.add_argument("--outputs-dir")
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--skip-v2-3", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    benchmark = json.loads((V3_ROOT / "benchmark.json").read_text())
    truth = {str(movie_id) for movie_id in benchmark["ground_truth_movie_ids"]}
    output_dir = (
        Path(args.outputs_dir)
        if args.outputs_dir
        else ROOT
        / "outputs"
        / f"threshold_sweep__{pair_slug(args.cheap_model, args.expensive_model)}"
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "benchmark_id": benchmark["benchmark_id"],
        "question": benchmark["question"],
        "cheap_model": args.cheap_model,
        "expensive_model": args.expensive_model,
        "thresholds": thresholds,
        "repetitions": args.repetitions,
        "cascade_target": args.cascade_target,
        "calibration_budget": args.calibration_budget,
        "cheap_batch_size": args.cheap_batch_size,
        "expensive_batch_size": args.expensive_batch_size,
        "v2_3_expensive_batch_size": args.v2_3_expensive_batch_size,
        "request_timeout": args.request_timeout,
        "dry_run": args.dry_run,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    python_path = Path(args.python)
    if python_path.is_absolute():
        python = str(python_path)
    else:
        resolved_python = shutil.which(args.python)
        python = resolved_python if resolved_python else str((Path.cwd() / python_path).resolve())
    selected = []
    if not args.skip_v2:
        selected.append("v2")
    if not args.skip_v2_3:
        selected.append("v2_3")

    rows: list[dict[str, object]] = []
    for implementation in selected:
        info = IMPLEMENTATIONS[implementation]
        for threshold in thresholds:
            run_dir = output_dir / implementation / threshold_slug(threshold)
            command = [
                python,
                str(V3_SCRIPTS / info["script"]),
                "--api-base",
                args.api_base,
                "--request-timeout",
                str(args.request_timeout),
                "--cheap-model",
                args.cheap_model,
                "--expensive-model",
                args.expensive_model,
                "--cascade-target",
                str(args.cascade_target),
                "--calibration-budget",
                str(args.calibration_budget),
                "--manual-confidence-threshold",
                format_threshold(threshold),
                *info["extra"](args),
                "--output-dir",
                str(run_dir),
            ]
            if args.dry_run:
                command.append("--dry-run")
            run_repeated(command, env, LAB_ROOT, args.repetitions, truth)
            run = json.loads((run_dir / "run_metrics.json").read_text())
            row = summarize_run(
                run,
                truth,
                implementation=implementation,
                label=info["label"],
                threshold=threshold,
            )
            rows.append(row)

    write_csv(output_dir / "threshold_metrics.csv", rows)
    write_summary(output_dir / "summary.md", benchmark, rows)
    plot_quality(rows, output_dir / "quality_metrics_vs_threshold.png")
    plot_final_rows(rows, output_dir / "final_rows_vs_threshold.png")
    print(f"Threshold sweep outputs: {output_dir}", flush=True)


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        threshold = float(item)
        if threshold < 0:
            raise ValueError("thresholds must be non-negative")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def summarize_run(
    run: dict,
    truth: set[str],
    implementation: str,
    label: str,
    threshold: float,
) -> dict[str, object]:
    found = {str(value) for value in run.get("found_movie_ids", [])}
    tp = len(found & truth)
    fp = len(found - truth)
    fn = len(truth - found)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if run.get("repetition_source") == "mean":
        tp = float(run.get("true_positives", tp))
        fp = float(run.get("false_positives", fp))
        fn = float(run.get("false_negatives", fn))
        precision = float(run.get("precision", precision))
        recall = float(run.get("recall", recall))
        f1 = float(run.get("f1", f1))
    return {
        "implementation": implementation,
        "label": label,
        "threshold": threshold,
        "repetitions": run.get("repetitions", 1),
        "cheap_model": run.get("cheap_model", ""),
        "expensive_model": run.get("expensive_model", ""),
        "wall_seconds": run.get("wall_seconds", 0.0),
        "llm_calls": run.get("llm_calls", 0),
        "cheap_calls": run.get("cheap_calls", 0),
        "expensive_calls": run.get("expensive_calls", 0),
        "cheap_early_accepts": run.get("cheap_early_accepts", 0),
        "cheap_early_rejects": run.get("cheap_early_rejects", 0),
        "expensive_candidates": run.get("expensive_candidates", 0),
        "final_answer_rows": run.get("final_answer_rows", 0),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, benchmark: dict, rows: list[dict[str, object]]) -> None:
    columns = [
        "label",
        "threshold",
        "precision",
        "recall",
        "f1",
        "final_answer_rows",
        "cheap_early_accepts",
        "cheap_early_rejects",
        "expensive_candidates",
    ]
    lines = [
        "# Cascade Threshold Sweep",
        "",
        f"Question: {benchmark['question']}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "Generated plots:",
        "",
        "- `quality_metrics_vs_threshold.png`",
        "- `final_rows_vs_threshold.png`",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_quality(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    metrics = [
        ("precision", "Precision", "#2878B5"),
        ("recall", "Recall", "#E07A1F"),
        ("f1", "F1", "#6A5ACD"),
    ]
    for axis, (metric, title, color) in zip(axes, metrics):
        for label, group in group_rows(rows).items():
            xs = [float(row["threshold"]) for row in group]
            ys = [float(row[metric]) for row in group]
            axis.plot(xs, ys, marker="o", linewidth=2, label=label, color=color if len(group_rows(rows)) == 1 else None)
        axis.set_title(title)
        axis.set_xlabel("Threshold")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_final_rows(rows: list[dict[str, object]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    for label, group in group_rows(rows).items():
        xs = [float(row["threshold"]) for row in group]
        ys = [float(row["final_answer_rows"]) for row in group]
        axis.plot(xs, ys, marker="o", linewidth=2, label=label)
    axis.set_title("Final Rows vs Threshold")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Mean final rows")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def group_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: float(row["threshold"]))
    return grouped


def threshold_slug(threshold: float) -> str:
    return "threshold_" + format_threshold(threshold).replace(".", "_")


def format_threshold(threshold: float) -> str:
    return f"{threshold:g}"


if __name__ == "__main__":
    main()
