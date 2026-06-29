#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from repetitions import run_repeated


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parent
DEFAULT_PYTHON = LAB_ROOT / "project SUQL" / ".venv" / "bin" / "python"


def model_slug(model: str) -> str:
    return model.removeprefix("ollama/").removesuffix(":latest").replace(":", "_").replace("/", "_")


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and evaluate both common-benchmark implementations.")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="ollama/gemma4:e4b")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument(
        "--trummer-request-timeout",
        type=float,
        default=3600,
        help="Timeout in seconds for each Trummer LLM request.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=9,
        help="Run each experiment this many times and aggregate numeric metrics by mean.",
    )
    parser.add_argument(
        "--skip-build-dataset",
        action="store_true",
        help="Use the already-synced common benchmark CSVs instead of rebuilding from the full SUQL dataset.",
    )
    parser.add_argument(
        "--skip-suql-baseline",
        action="store_true",
        help="Keep an existing SUQL result and run only Trummer before regenerating comparison artifacts.",
    )
    parser.add_argument(
        "--skip-trummer",
        action="store_true",
        help="Keep an existing Trummer result and run only SUQL before regenerating comparison artifacts.",
    )
    args = parser.parse_args()

    python_path = Path(args.python).expanduser()
    if not python_path.is_absolute():
        python_path = (LAB_ROOT / python_path).absolute()
    python = str(python_path)
    experiment_dir = ROOT / "outputs" / model_slug(args.model)
    benchmark = json.loads((ROOT / "benchmark.json").read_text())
    truth = {str(movie_id) for movie_id in benchmark["ground_truth_movie_ids"]}
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    env.setdefault("MPLBACKEND", "Agg")
    commands = []
    if not args.skip_build_dataset:
        commands.append([python, str(ROOT / "scripts" / "build_dataset.py")])
    if not args.skip_suql_baseline:
        commands.append(
            [
            python,
            str(ROOT / "scripts" / "run_suql_baseline.py"),
            "--api-base",
            args.api_base,
            "--model",
            args.model,
            "--output-dir",
            str(experiment_dir / "suql_baseline"),
            ]
        )
    if not args.skip_trummer:
        commands.append(
            [
            python,
            str(ROOT / "scripts" / "run_trummer.py"),
            "--api-base",
            args.api_base,
            "--model",
            args.model,
            "--output-dir",
            str(experiment_dir / "trummer_heterogen_v1"),
            "--request-timeout",
            str(args.trummer_request_timeout),
            ]
        )
    if args.dry_run:
        for command in commands:
            if command[1].endswith(("run_suql_baseline.py", "run_trummer.py")):
                command.append("--dry-run")
    for command in commands:
        run_repeated(command, env, ROOT, args.repetitions, truth)
    run(
        [
            python,
            str(ROOT / "scripts" / "evaluate_and_plot.py"),
            "--outputs-dir",
            str(experiment_dir),
        ],
        env,
    )


if __name__ == "__main__":
    main()
