#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_ids(path: Path, column: str = "movie_id") -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            str(row[column])
            for row in csv.DictReader(handle)
            if str(row.get(column, "")).strip()
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate per-question found_rows.csv files against "
            "common_benchmark_three_questions."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory containing question_*/found_rows.csv.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    rows = []
    total_tp = total_fp = total_fn = 0

    for item in manifest["questions"]:
        question_dir = item["directory"]
        truth = read_ids(ROOT / question_dir / "data" / "ground_truth.csv")
        prediction_path = predictions_dir / question_dir / "found_rows.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing predictions: {prediction_path}")
        found = read_ids(prediction_path)
        tp = len(found & truth)
        fp = len(found - truth)
        fn = len(truth - found)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "question": question_dir,
                "difficulty": item["difficulty"],
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    summary = {
        "questions": rows,
        "macro_f1": sum(row["f1"] for row in rows) / len(rows),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }

    output = Path(args.output) if args.output else predictions_dir / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
