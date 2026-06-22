#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import ROOT, pair_slug


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Trummer heterogen_v1 and heterogen_v2.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="ollama/gemma2:2b")
    parser.add_argument("--expensive-model", default="ollama/qwen2.5:3b")
    parser.add_argument("--cheap-accept-threshold", type=float, default=3.0)
    parser.add_argument("--cheap-reject-threshold", type=float, default=-1.5)
    parser.add_argument("--expensive-batch-size", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-v1", action="store_true")
    parser.add_argument("--skip-v2", action="store_true")
    args = parser.parse_args()

    experiment = ROOT / "outputs" / pair_slug(args.cheap_model, args.expensive_model)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    common = ["--api-base", args.api_base]
    commands = []
    if not args.skip_v1:
        commands.append([
            args.python, str(ROOT / "scripts" / "run_heterogen_v1.py"), *common,
            "--model", args.expensive_model,
            "--request-timeout", str(args.request_timeout),
            "--output-dir", str(experiment / "trummer_heterogen_v1"),
        ])
    if not args.skip_v2:
        commands.append([
            args.python, str(ROOT / "scripts" / "run_heterogen_v2.py"), *common,
            "--cheap-model", args.cheap_model,
            "--expensive-model", args.expensive_model,
            "--cheap-accept-threshold", str(args.cheap_accept_threshold),
            "--cheap-reject-threshold", str(args.cheap_reject_threshold),
            "--expensive-batch-size", str(args.expensive_batch_size),
            "--request-timeout", str(args.request_timeout),
            "--output-dir", str(experiment / "trummer_heterogen_v2_cascade"),
        ])
    if args.dry_run:
        for command in commands:
            command.append("--dry-run")
    for command in commands:
        run(command, env)
    run([args.python, str(ROOT / "scripts" / "evaluate_and_plot.py"), "--outputs-dir", str(experiment)], env)


if __name__ == "__main__":
    main()
