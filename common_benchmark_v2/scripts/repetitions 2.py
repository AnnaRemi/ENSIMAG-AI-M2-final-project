from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def run_repeated(
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    repetitions: int,
    truth: set[str],
) -> None:
    output_dir = _output_dir(command)
    if output_dir is None or repetitions <= 1:
        _run(command, env, cwd)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    repetitions_dir = output_dir / "repetitions"
    repetitions_dir.mkdir(parents=True, exist_ok=True)
    run_metrics: list[dict] = []
    last_run_dir: Path | None = None

    for index in range(1, repetitions + 1):
        run_dir = repetitions_dir / f"run_{index:02d}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        repeated = _with_output_dir(command, run_dir)
        print(
            f"[repetition {index}/{repetitions}] {' '.join(command[:2])} -> {run_dir}",
            flush=True,
        )
        _run(repeated, env, cwd)
        run_metrics.append(json.loads((run_dir / "run_metrics.json").read_text()))
        last_run_dir = run_dir

    if last_run_dir is None:
        return
    _copy_last_run_outputs(last_run_dir, output_dir)
    aggregate = _mean_metrics(run_metrics, truth)
    aggregate["repetitions"] = repetitions
    aggregate["repetition_metrics_file"] = "run_metrics_repetitions.csv"
    aggregate["repetition_source"] = "mean"
    (output_dir / "run_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    _write_repetition_csv(output_dir / "run_metrics_repetitions.csv", run_metrics, truth)


def _run(command: list[str], env: dict[str, str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _output_dir(command: list[str]) -> Path | None:
    try:
        index = command.index("--output-dir")
    except ValueError:
        return None
    return Path(command[index + 1])


def _with_output_dir(command: list[str], output_dir: Path) -> list[str]:
    repeated = list(command)
    index = repeated.index("--output-dir")
    repeated[index + 1] = str(output_dir)
    return repeated


def _copy_last_run_outputs(source: Path, target: Path) -> None:
    for path in target.iterdir():
        if path.name == "repetitions":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for path in source.iterdir():
        destination = target / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)


def _mean_metrics(runs: list[dict], truth: set[str]) -> dict:
    result = dict(runs[-1])
    numeric_keys = sorted(
        {
            key
            for run in runs
            for key, value in run.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for key in numeric_keys:
        values = [
            float(run[key])
            for run in runs
            if isinstance(run.get(key), (int, float)) and not isinstance(run.get(key), bool)
        ]
        if values:
            result[key] = sum(values) / len(values)
    qualities = [_quality(run, truth) for run in runs]
    for key in ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1"):
        result[key] = sum(float(item[key]) for item in qualities) / len(qualities)
    return result


def _write_repetition_csv(path: Path, runs: Iterable[dict], truth: set[str]) -> None:
    rows = []
    for index, run in enumerate(runs, 1):
        quality = _quality(run, truth)
        row = {
            "repetition": index,
            "implementation": run.get("implementation", ""),
            "mode": run.get("mode", ""),
            "model": run.get("model", ""),
            "wall_seconds": run.get("wall_seconds", 0.0),
            "engine_seconds": run.get("engine_seconds", 0.0),
            "cpu_seconds": run.get("cpu_seconds", 0.0),
            "llm_calls": run.get("llm_calls", 0),
            "final_answer_rows": run.get("final_answer_rows", 0),
            **quality,
        }
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _quality(run: dict, truth: set[str]) -> dict[str, float]:
    found = {str(value) for value in run.get("found_movie_ids", [])}
    tp = len(found & truth)
    fp = len(found - truth)
    fn = len(truth - found)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
