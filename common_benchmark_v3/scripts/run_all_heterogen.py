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
        description="Run the Heterogen versions on the one-question dataset."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="ollama/gemma4:e2b")
    parser.add_argument("--expensive-model", default="ollama/gemma4:e4b")
    parser.add_argument("--structured-parser-model")
    parser.add_argument("--disable-llm-structured-parser", action="store_true")
    parser.add_argument("--cascade-target", type=float, default=0.9)
    parser.add_argument("--calibration-budget", type=int, default=20)
    parser.add_argument("--v2-manual-confidence-threshold", type=float)
    parser.add_argument("--cheap-batch-size", type=int, default=8)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--v2-3-expensive-batch-size", type=int, default=32)
    parser.add_argument("--max-expensive-calls", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--token-threshold", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--max-movie-block-size", type=int, default=25)
    parser.add_argument("--max-review-block-size", type=int, default=8)
    parser.add_argument("--outputs-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=9,
        help="Run each experiment this many times and aggregate numeric metrics by mean.",
    )
    parser.add_argument("--skip-v1", action="store_true")
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--skip-v2-2", action="store_true")
    parser.add_argument("--skip-v2-3", action="store_true")
    parser.add_argument("--skip-v3", action="store_true")
    args = parser.parse_args()
    structured_parser_model = args.structured_parser_model or args.cheap_model

    python_path = Path(args.python)
    if not python_path.is_absolute():
        python_path = Path.cwd() / python_path
    python = str(python_path)
    output_dir = (
        Path(args.outputs_dir)
        if args.outputs_dir
        else ROOT
        / "outputs"
        / f"all_heterogen__{pair_slug(args.cheap_model, args.expensive_model)}"
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads((ROOT / "benchmark.json").read_text())
    experiment_config = {
        "benchmark_id": benchmark["benchmark_id"],
        "question": benchmark["question"],
        "implementations": [
            "heterogen_v1",
            "heterogen_v2",
            "heterogen_v2_2",
            "heterogen_v2_3",
            "heterogen_v3",
        ],
        "cheap_model": args.cheap_model,
        "expensive_model": args.expensive_model,
        "structured_parser_model": structured_parser_model,
        "disable_llm_structured_parser": args.disable_llm_structured_parser,
        "cascade_target": args.cascade_target,
        "calibration_budget": args.calibration_budget,
        "v2_manual_confidence_threshold": args.v2_manual_confidence_threshold,
        "cheap_batch_size": args.cheap_batch_size,
        "expensive_batch_size": args.expensive_batch_size,
        "v2_3_expensive_batch_size": args.v2_3_expensive_batch_size,
        "max_expensive_calls": args.max_expensive_calls,
        "parallel_workers": args.parallel_workers,
        "request_timeout": args.request_timeout,
        "token_threshold": args.token_threshold,
        "max_completion_tokens": args.max_completion_tokens,
        "max_movie_block_size": args.max_movie_block_size,
        "max_review_block_size": args.max_review_block_size,
        "repetitions": args.repetitions,
        "dry_run": args.dry_run,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2) + "\n"
    )

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    truth = {str(movie_id) for movie_id in benchmark["ground_truth_movie_ids"]}
    common = [
        "--api-base",
        args.api_base,
        "--request-timeout",
        str(args.request_timeout),
    ]
    block_common = [
        "--token-threshold",
        str(args.token_threshold),
        "--max-completion-tokens",
        str(args.max_completion_tokens),
        "--max-movie-block-size",
        str(args.max_movie_block_size),
        "--max-review-block-size",
        str(args.max_review_block_size),
    ]
    structured_parser_common = [
        "--structured-parser-model",
        structured_parser_model,
        *(
            ["--disable-llm-structured-parser"]
            if args.disable_llm_structured_parser
            else []
        ),
    ]
    cascade_common = [
        "--cheap-model",
        args.cheap_model,
        "--expensive-model",
        args.expensive_model,
        "--cascade-target",
        str(args.cascade_target),
        "--calibration-budget",
        str(args.calibration_budget),
    ]
    commands: list[tuple[str, list[str]]] = [
        (
            "v1",
            [
            python,
            str(ROOT / "scripts" / "run_heterogen_v1.py"),
            *common,
            *block_common,
            "--model",
            args.expensive_model,
            "--output-dir",
            str(output_dir / "heterogen_v1"),
            ],
        ),
        (
            "v2",
            [
            python,
            str(ROOT / "scripts" / "run_heterogen_v2.py"),
            *common,
            *cascade_common,
            *(
                [
                    "--manual-confidence-threshold",
                    str(args.v2_manual_confidence_threshold),
                ]
                if args.v2_manual_confidence_threshold is not None
                else []
            ),
            "--expensive-batch-size",
            str(args.expensive_batch_size),
            "--output-dir",
            str(output_dir / "heterogen_v2"),
            ],
        ),
        (
            "v2_2",
            [
            python,
            str(ROOT / "scripts" / "run_heterogen_v2_2.py"),
            *common,
            *structured_parser_common,
            *block_common,
            "--model",
            args.expensive_model,
            "--output-dir",
            str(output_dir / "heterogen_v2_2"),
            ],
        ),
        (
            "v2_3",
            [
            python,
            str(ROOT / "scripts" / "run_heterogen_v2_3.py"),
            *common,
            *cascade_common,
            "--cheap-batch-size",
            str(args.cheap_batch_size),
            "--expensive-batch-size",
            str(args.v2_3_expensive_batch_size),
            "--output-dir",
            str(output_dir / "heterogen_v2_3"),
            ],
        ),
        (
            "v3",
            [
            python,
            str(ROOT / "scripts" / "run_heterogen_v3.py"),
            *common,
            *structured_parser_common,
            *cascade_common,
            "--expensive-batch-size",
            str(args.expensive_batch_size),
            "--max-expensive-calls",
            str(args.max_expensive_calls),
            "--output-dir",
            str(output_dir / "heterogen_v3"),
            ],
        ),
    ]
    skipped = {
        "v1": args.skip_v1,
        "v2": args.skip_v2,
        "v2_2": args.skip_v2_2,
        "v2_3": args.skip_v2_3,
        "v3": args.skip_v3,
    }
    commands = [
        (name, command)
        for name, command in commands
        if not skipped[name]
    ]
    if args.dry_run:
        for _, command in commands:
            command.append("--dry-run")
    for _, command in commands:
        run_repeated(command, env, ROOT, args.repetitions, truth)
    evaluation_command = [
        python,
        str(ROOT / "scripts" / "evaluate_all_heterogen.py"),
        "--outputs-dir",
        str(output_dir),
    ]
    if any(skipped.values()):
        evaluation_command.append("--allow-missing")
    run(evaluation_command, env)
    print(f"All-Heterogen outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
