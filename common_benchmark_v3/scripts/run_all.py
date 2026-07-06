#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import ROOT, pair_slug
from repetitions import run_repeated


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SUQL baseline and Trummer heterogen variants."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="ollama/gemma4:e2b")
    parser.add_argument("--expensive-model", default="ollama/gemma4:e4b")
    parser.add_argument("--structured-parser-model")
    parser.add_argument("--disable-llm-structured-parser", action="store_true")
    parser.add_argument("--cascade-target", type=float, default=0.9)
    parser.add_argument("--calibration-budget", type=int, default=20)
    parser.add_argument("--cheap-batch-size", type=int, default=8)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--v2-3-expensive-batch-size", type=int, default=32)
    parser.add_argument("--max-expensive-calls", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=9,
        help="Run each experiment this many times and aggregate numeric metrics by mean.",
    )
    parser.add_argument("--skip-suql", action="store_true")
    parser.add_argument("--skip-v1", action="store_true")
    parser.add_argument("--skip-v2-2", action="store_true")
    parser.add_argument("--skip-v2-3", action="store_true")
    parser.add_argument("--skip-v3", action="store_true")
    parser.add_argument("--skip-v2", action="store_true")
    args = parser.parse_args()
    structured_parser_model = args.structured_parser_model or args.cheap_model
    python_path = Path(args.python)
    if not python_path.is_absolute():
        python_path = Path.cwd() / python_path
    args.python = str(python_path)

    experiment = Path(args.outputs_dir) if args.outputs_dir else ROOT / "outputs" / pair_slug(args.cheap_model, args.expensive_model)
    if not experiment.is_absolute():
        experiment = ROOT / experiment
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    benchmark = json.loads((ROOT / "benchmark.json").read_text())
    truth = {str(movie_id) for movie_id in benchmark["ground_truth_movie_ids"]}
    common = ["--api-base", args.api_base]
    structured_parser_common = [
        "--structured-parser-model",
        structured_parser_model,
        *(
            ["--disable-llm-structured-parser"]
            if args.disable_llm_structured_parser
            else []
        ),
    ]
    commands = []
    if not args.skip_suql:
        commands.append([
            args.python, str(ROOT / "scripts" / "run_suql_baseline.py"), *common,
            "--model", args.expensive_model,
            "--output-dir", str(experiment / "suql_baseline"),
        ])
    if not args.skip_v1:
        commands.append([
            args.python, str(ROOT / "scripts" / "run_heterogen_v1.py"), *common,
            "--model", args.expensive_model,
            "--request-timeout", str(args.request_timeout),
            "--output-dir", str(experiment / "trummer_heterogen_v1"),
        ])
    if not args.skip_v2_2:
        commands.append([
            args.python, str(ROOT / "scripts" / "run_heterogen_v2_2.py"), *common,
            *structured_parser_common,
            "--model", args.expensive_model,
            "--request-timeout", str(args.request_timeout),
            "--output-dir", str(experiment / "trummer_heterogen_v2_2_structured_pruned"),
        ])
    if not args.skip_v3:
        commands.append([
            args.python, str(ROOT / "scripts" / "run_heterogen_v3.py"), *common,
            *structured_parser_common,
            "--cheap-model", args.cheap_model,
            "--expensive-model", args.expensive_model,
            "--cascade-target", str(args.cascade_target),
            "--calibration-budget", str(args.calibration_budget),
            "--expensive-batch-size", str(args.expensive_batch_size),
            "--max-expensive-calls", str(args.max_expensive_calls),
            "--request-timeout", str(args.request_timeout),
            "--output-dir", str(experiment / "trummer_heterogen_v3_pruned_cascade"),
        ])
    if not args.skip_v2_3:
        commands.append([
            args.python, str(ROOT / "scripts" / "run_heterogen_v2_3.py"), *common,
            "--cheap-model", args.cheap_model,
            "--expensive-model", args.expensive_model,
            "--cascade-target", str(args.cascade_target),
            "--calibration-budget", str(args.calibration_budget),
            "--cheap-batch-size", str(args.cheap_batch_size),
            "--expensive-batch-size", str(args.v2_3_expensive_batch_size),
            "--request-timeout", str(args.request_timeout),
            "--output-dir", str(experiment / "trummer_heterogen_v2_3_batched_cascade"),
        ])
    if not args.skip_v2:
        commands.append([
            args.python, str(ROOT / "scripts" / "run_heterogen_v2.py"), *common,
            "--cheap-model", args.cheap_model,
            "--expensive-model", args.expensive_model,
            "--cascade-target", str(args.cascade_target),
            "--calibration-budget", str(args.calibration_budget),
            "--expensive-batch-size", str(args.expensive_batch_size),
            "--request-timeout", str(args.request_timeout),
            "--output-dir", str(experiment / "trummer_heterogen_v2_cascade"),
        ])
    if args.dry_run:
        for command in commands:
            command.append("--dry-run")
    for command in commands:
        run_repeated(command, env, ROOT, args.repetitions, truth)
    run([args.python, str(ROOT / "scripts" / "evaluate_and_plot.py"), "--outputs-dir", str(experiment)], env)


if __name__ == "__main__":
    main()
