#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import ROOT
from repetitions import run_repeated


METHOD_DIRS = {
    "suql": "suql_baseline",
    "v2_3": "heterogen_v2_3",
    "v3": "heterogen_v3",
    "v3_2": "heterogen_v3_2",
}


def question_dirs() -> list[str]:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    questions = manifest.get("questions", [])
    if len(questions) != 10:
        raise RuntimeError(f"Expected 10 questions in manifest, found {len(questions)}")
    return [str(item["directory"]) for item in questions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="ollama/gemma4:e2b")
    parser.add_argument("--expensive-model", default="ollama/gemma4:e4b")
    parser.add_argument("--structured-parser-model")
    parser.add_argument("--disable-llm-structured-parser", action="store_true")
    parser.add_argument("--cascade-target", type=float, default=0.9)
    parser.add_argument("--calibration-budget", type=int, default=20)
    parser.add_argument("--manual-confidence-threshold", type=float)
    parser.add_argument("--cheap-batch-size", type=int, default=8)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--v2-3-expensive-batch-size", type=int, default=32)
    parser.add_argument("--max-expensive-calls", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--token-threshold", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--max-movie-block-size", type=int, default=25)
    parser.add_argument("--max-review-block-size", type=int, default=8)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=11,
        help="Run each question/method this many times and aggregate numeric metrics by mean.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHOD_DIRS),
        default=list(METHOD_DIRS),
        help="Methods to run. Defaults to SUQL, V2_3, V3, and V3_2.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    structured_parser_model = args.structured_parser_model or args.cheap_model
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

    for question_dir in question_dirs():
        spec = json.loads((ROOT / question_dir / "benchmark.json").read_text())
        truth = set(spec["ground_truth_movie_ids"])
        if not truth:
            raise RuntimeError(f"{question_dir} has empty ground truth")
        for method in args.methods:
            command = [
                args.python,
                str(ROOT / "scripts" / "run_method.py"),
                "--method",
                method,
                "--question-dir",
                question_dir,
                "--api-base",
                args.api_base,
                "--cheap-model",
                args.cheap_model,
                "--expensive-model",
                args.expensive_model,
                "--structured-parser-model",
                structured_parser_model,
                *(
                    ["--disable-llm-structured-parser"]
                    if args.disable_llm_structured_parser
                    else []
                ),
                "--cascade-target",
                str(args.cascade_target),
                "--calibration-budget",
                str(args.calibration_budget),
                *(
                    ["--manual-confidence-threshold", str(args.manual_confidence_threshold)]
                    if args.manual_confidence_threshold is not None
                    else []
                ),
                "--cheap-batch-size",
                str(args.cheap_batch_size),
                "--expensive-batch-size",
                str(args.expensive_batch_size),
                "--v2-3-expensive-batch-size",
                str(args.v2_3_expensive_batch_size),
                "--max-expensive-calls",
                str(args.max_expensive_calls),
                "--request-timeout",
                str(args.request_timeout),
                "--token-threshold",
                str(args.token_threshold),
                "--max-completion-tokens",
                str(args.max_completion_tokens),
                "--max-movie-block-size",
                str(args.max_movie_block_size),
                "--max-review-block-size",
                str(args.max_review_block_size),
                "--output-dir",
                str(output_dir / question_dir / METHOD_DIRS[method]),
            ]
            run_repeated(command, env, ROOT, args.repetitions, truth)

    subprocess.run(
        [
            args.python,
            str(ROOT / "scripts" / "evaluate_and_plot.py"),
            "--outputs-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
