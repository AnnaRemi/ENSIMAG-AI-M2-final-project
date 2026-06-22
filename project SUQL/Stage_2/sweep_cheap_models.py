#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

try:
    from Stage_1.profiler import OllamaLogOddsScorer
except ImportError:  # pragma: no cover
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from Stage_1.profiler import OllamaLogOddsScorer


STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parent
SWEEP_DIR = STAGE_DIR / "model_sweeps"

DEFAULT_MODELS = [
    "gemma2:2b",
    "llama3.2:1b",
    "smollm2:360m",
    "smollm2:1.7b",
    "tinyllama:1.1b",
]

SMOKE_EXAMPLES = [
    (
        "This movie is funny, witty, and consistently entertaining.",
        "Does the reviewer describe the movie as funny, humorous, witty, or entertaining?",
    ),
    (
        "The movie is dull, slow, and not exciting at all.",
        "Does the reviewer describe the movie as exciting, thrilling, intense, energetic, or action-packed?",
    ),
]


def litellm_model_name(model: str) -> str:
    return model if model.startswith("ollama/") else f"ollama/{model}"


def ollama_model_name(model: str) -> str:
    return model.removeprefix("ollama/")


def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ollama_model_name(model)).strip("_")


def installed_models(api_base: str) -> set[str]:
    try:
        response = httpx.get(f"{api_base.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {api_base.rstrip('/')}. "
            "Start Ollama or set SUQL_API_BASE/--api-base to the reachable server."
        ) from exc
    payload = response.json()
    names = set()
    for item in payload.get("models", []):
        name = item.get("name")
        if name:
            names.add(str(name))
    return names


def smoke_score_model(model: str, api_base: str) -> dict[str, str | float]:
    scorer = OllamaLogOddsScorer(model=litellm_model_name(model), api_base=api_base, timeout=120)
    scores: list[float] = []
    for review, question in SMOKE_EXAMPLES:
        scores.append(float(scorer.score(review, question)))
    return {
        "status": "ok",
        "score_positive_example": scores[0],
        "score_negative_example": scores[1],
        "error": "",
    }


def run_benchmark_for_model(
    model: str,
    api_base: str,
    expensive_model: str,
    sample_size: int,
    seed: int,
    python: str,
    run_prefix: str,
) -> tuple[int, str, str]:
    run_name = f"{run_prefix}_{safe_name(model)}"
    log_path = STAGE_DIR / f"{run_name}.log"
    cmd = [
        python,
        str(STAGE_DIR / "benchmark_stage2.py"),
        "--sample-size",
        str(sample_size),
        "--seed",
        str(seed),
        "--api-base",
        api_base,
        "--model",
        expensive_model,
        "--cheap-model",
        litellm_model_name(model),
        "--run-name",
        run_name,
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, run_name, f"{elapsed:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test and optionally benchmark cheap Ollama models for Stage 2.")
    parser.add_argument("--api-base", default=os.environ.get("SUQL_API_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--installed-only", action="store_true", help="Skip candidates not listed by /api/tags.")
    parser.add_argument("--run-benchmarks", action="store_true", help="Run Stage 2 benchmark for models that pass smoke scoring.")
    parser.add_argument("--expensive-model", default=os.environ.get("SUQL_EXPENSIVE_MODEL", os.environ.get("SUQL_MODEL", "ollama/phi4-mini")))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-prefix", default=f"cheap_sweep_{time.strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SWEEP_DIR / f"{args.run_prefix}.csv"

    available: set[str] | None = None
    if args.installed_only:
        try:
            available = installed_models(args.api_base)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print(
                "\nExamples:\n"
                "  ollama serve\n"
                "  export SUQL_API_BASE=http://127.0.0.1:11434\n"
                "  python Stage_2/sweep_cheap_models.py --api-base \"$SUQL_API_BASE\" --models gemma2:2b\n",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc

    rows: list[dict[str, str | float]] = []
    for model in args.models:
        ollama_name = ollama_model_name(model)
        row: dict[str, str | float] = {
            "cheap_model": litellm_model_name(model),
            "ollama_model": ollama_name,
            "installed": "",
            "status": "",
            "score_positive_example": "",
            "score_negative_example": "",
            "error": "",
            "benchmark_run": "",
            "benchmark_exit_code": "",
            "benchmark_seconds": "",
        }

        if available is not None:
            is_installed = ollama_name in available
            row["installed"] = str(is_installed)
            if not is_installed:
                row["status"] = "skipped_not_installed"
                rows.append(row)
                continue

        try:
            row.update(smoke_score_model(model, args.api_base))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"[:500]
            rows.append(row)
            continue

        if args.run_benchmarks:
            code, run_name, seconds = run_benchmark_for_model(
                model=model,
                api_base=args.api_base,
                expensive_model=args.expensive_model,
                sample_size=args.sample_size,
                seed=args.seed,
                python=args.python,
                run_prefix=args.run_prefix,
            )
            row["benchmark_run"] = run_name
            row["benchmark_exit_code"] = str(code)
            row["benchmark_seconds"] = seconds

        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sweep saved to: {output_path}")
    for row in rows:
        print(
            f"{row['cheap_model']}: {row['status']} "
            f"pos={row['score_positive_example']} neg={row['score_negative_example']} "
            f"run={row['benchmark_run']} error={row['error']}"
        )


if __name__ == "__main__":
    main()
