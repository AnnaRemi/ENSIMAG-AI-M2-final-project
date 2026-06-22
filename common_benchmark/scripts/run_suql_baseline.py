#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parent
SUQL_ROOT = LAB_ROOT / "project SUQL"
ENGINE_DIR = Path(
    os.environ.get("COMMON_BENCHMARK_SUQL_ENGINE_DIR", SUQL_ROOT / "src_baseline")
)


def model_slug(model: str) -> str:
    return model.removeprefix("ollama/").removesuffix(":latest").replace(":", "_").replace("/", "_")


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SUQL baseline on the common benchmark.")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="ollama/gemma2:2b")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and query without LLM execution.")
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir or ROOT / "outputs" / model_slug(args.model) / "suql_baseline"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads((ROOT / "benchmark.json").read_text())

    if args.dry_run:
        import pandas as pd

        rows = pd.read_csv(ROOT / "data" / "imdb_joined.csv")
        results = rows[
            rows["movie_id"].isin(benchmark["ground_truth_movie_ids"])
        ][["movie_id", "title", "year", "runtime", "director", "genres"]].copy()
        results.to_csv(output_dir / "found_rows.csv", index=False)
        payload = {
            "implementation": "suql_baseline",
            "mode": "dry_run_ground_truth_wiring",
            "model": args.model,
            "api_base": args.api_base,
            "cpu_seconds": 0.0,
            "engine_seconds": 0.0,
            "wall_seconds": 0.0,
            "llm_calls": 0,
            "final_answer_rows": int(len(results)),
            "found_movie_ids": sorted(results["movie_id"].astype(str).unique()),
        }
        (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    metrics_path = output_dir / "engine_metrics.json"
    os.environ["SUQL_DATA_PATH"] = str(ROOT / "data" / "imdb_joined.csv")
    os.environ["SUQL_API_BASE"] = args.api_base
    os.environ["SUQL_MODEL"] = args.model
    os.environ["SUQL_METRICS_PATH"] = str(metrics_path)
    sys.path.insert(0, str(ENGINE_DIR))

    import suql_engine

    suql_engine._answer_cache.clear()
    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    results = suql_engine.ask_with_suql(
        benchmark["suql_query"],
        output_csv=str(output_dir / "found_rows.csv"),
        verbose=True,
    )
    wall_seconds = time.perf_counter() - started_wall
    process_cpu_seconds = cpu_seconds() - started_cpu
    engine_metrics = json.loads(metrics_path.read_text())

    payload = {
        "implementation": "suql_baseline",
        "mode": "llm",
        "model": args.model,
        "api_base": args.api_base,
        "cpu_seconds": process_cpu_seconds,
        "engine_seconds": float(engine_metrics["engine_seconds"]),
        "wall_seconds": wall_seconds,
        "llm_calls": int(engine_metrics["llm_prompts_issued"]),
        "final_answer_rows": int(len(results)),
        "found_movie_ids": sorted(results["movie_id"].astype(str).unique()),
        "structured_candidates": int(engine_metrics["structured_candidates"]),
    }
    (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
