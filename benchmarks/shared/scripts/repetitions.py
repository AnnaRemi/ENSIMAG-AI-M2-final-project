from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, TextIO


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
    prepared: list[tuple[int, Path, list[str]]] = []
    for index in range(1, repetitions + 1):
        run_dir = repetitions_dir / f"run_{index:02d}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        repeated = _with_output_dir(command, run_dir)
        print(
            f"[repetition {index}/{repetitions}] {' '.join(command[:2])} -> {run_dir}",
            flush=True,
        )
        prepared.append((index, run_dir, repeated))

    workers = max(1, min(repetitions, int(os.environ.get("PARALLEL_WORKERS", "1"))))
    if workers == 1:
        for _index, _run_dir, repeated in prepared:
            _run(repeated, env, cwd)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_run, repeated, env, cwd)
                for _index, _run_dir, repeated in prepared
            ]
            for future in futures:
                future.result()

    run_metrics = [
        json.loads((run_dir / "run_metrics.json").read_text())
        for _index, run_dir, _repeated in prepared
    ]
    last_run_dir = prepared[-1][1] if prepared else None

    if last_run_dir is None:
        return
    _copy_last_run_outputs(last_run_dir, output_dir)
    aggregate = _mean_metrics(run_metrics, truth)
    aggregate["repetitions"] = repetitions
    aggregate["repetition_metrics_file"] = "run_metrics_repetitions.csv"
    aggregate["repetition_source"] = "mean"
    (output_dir / "run_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    _write_repetition_csv(output_dir / "run_metrics_repetitions.csv", run_metrics, truth)
    shutil.rmtree(repetitions_dir)


def _run(command: list[str], env: dict[str, str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    output_dir = _output_dir(command)
    if output_dir is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stdout.log").open("w") as stdout_log, (
        output_dir / "stderr.log"
    ).open("w") as stderr_log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=_tee_lines,
            args=(process.stdout, sys.stdout, stdout_log),
        )
        stderr_thread = threading.Thread(
            target=_tee_lines,
            args=(process.stderr, sys.stderr, stderr_log),
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
    if returncode:
        _print_failure_tail(output_dir)
        raise subprocess.CalledProcessError(returncode, command)


def _tee_lines(source: TextIO, console: TextIO, log_file: TextIO) -> None:
    for line in source:
        console.write(line)
        console.flush()
        log_file.write(line)
        log_file.flush()


def _print_failure_tail(output_dir: Path, lines: int = 80) -> None:
    print(f"Child command failed; run directory: {output_dir}", file=sys.stderr, flush=True)
    for name in ("stderr.log", "stdout.log"):
        path = output_dir / name
        print(f"--- tail -{lines} {path} ---", file=sys.stderr, flush=True)
        if not path.exists():
            print("(missing)", file=sys.stderr, flush=True)
            continue
        tail = path.read_text(errors="replace").splitlines()[-lines:]
        for line in tail:
            print(line, file=sys.stderr, flush=True)


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
    for key in (
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
    ):
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
            "cheap_calls": run.get("cheap_calls", 0),
            "expensive_calls": run.get("expensive_calls", 0),
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
