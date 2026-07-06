#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import ROOT

sys.path.insert(0, str(ROOT.parent / "common_benchmark_v3" / "scripts"))
from repetitions import run_repeated


METHOD_DIRS = {
    "suql": "suql_baseline",
    "v2_2": "heterogen_v2_2",
    "v2_3": "heterogen_v2_3",
    "v3": "heterogen_v3",
    "v3_2": "heterogen_v3_2",
}


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
    parser.add_argument("--cheap-accept-threshold", type=float, default=3.0)
    parser.add_argument("--cheap-reject-threshold", type=float, default=-1.5)
    parser.add_argument("--cheap-batch-size", type=int, default=8)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--v2-3-expensive-batch-size", type=int, default=32)
    parser.add_argument("--max-expensive-calls", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=600)
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
        help="Methods to run. Defaults to all methods.",
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
    for question_dir in ("question_1_easy", "question_2_medium", "question_3_hard"):
        truth = set(
            json.loads((ROOT / question_dir / "benchmark.json").read_text())[
                "ground_truth_movie_ids"
            ]
        )
        for method in args.methods:
            method_dir = METHOD_DIRS[method]
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
                "--cheap-accept-threshold",
                str(args.cheap_accept_threshold),
                "--cheap-reject-threshold",
                str(args.cheap_reject_threshold),
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
                "--output-dir",
                str(output_dir / question_dir / method_dir),
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
