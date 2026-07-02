#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import ROOT


METHOD_DIRS = {
    "suql": "suql_baseline",
    "v2_2": "heterogen_v2_2",
    "v2_3": "heterogen_v2_3",
    "v3": "heterogen_v3",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="ollama/gemma4:e2b")
    parser.add_argument("--expensive-model", default="ollama/gemma4:e4b")
    parser.add_argument("--cheap-accept-threshold", type=float, default=3.0)
    parser.add_argument("--cheap-reject-threshold", type=float, default=-1.5)
    parser.add_argument("--cheap-batch-size", type=int, default=8)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--v2-3-expensive-batch-size", type=int, default=32)
    parser.add_argument("--max-expensive-calls", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    for question_dir in ("question_1_easy", "question_2_medium", "question_3_hard"):
        for method, method_dir in METHOD_DIRS.items():
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
            print("+", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
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
