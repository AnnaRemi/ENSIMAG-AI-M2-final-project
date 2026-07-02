#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from common import LAB_ROOT, ROOT, benchmark, cpu_seconds


SUQL_ROOT = LAB_ROOT / "project SUQL"
ENGINE_DIR = Path(
    os.environ.get("COMMON_BENCHMARK_V3_SUQL_ENGINE_DIR", SUQL_ROOT / "src_baseline")
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SUQL baseline for common benchmark v3.")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="ollama/gemma4:e4b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = benchmark()

    if args.dry_run:
        rows = pd.read_csv(ROOT / "data" / "imdb_joined.csv")
        results = rows[
            rows["movie_id"].astype(str).isin(spec["ground_truth_movie_ids"])
        ][["movie_id", "title", "year", "runtime", "director", "genres"]].copy()
        results.to_csv(output_dir / "found_rows.csv", index=False)
        payload = {
            "implementation": "suql_baseline",
            "mode": "dry_run_ground_truth_wiring",
            "model": args.model,
            "cheap_model": "",
            "expensive_model": args.model,
            "cpu_seconds": 0.0,
            "engine_seconds": 0.0,
            "wall_seconds": 0.0,
            "llm_calls": 0,
            "block_join_calls": 0,
            "cheap_calls": 0,
            "expensive_calls": 0,
            "final_answer_rows": int(len(results)),
            "found_movie_ids": sorted(results["movie_id"].astype(str).unique()),
        }
        (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)
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
        spec["suql_query"],
        output_csv=str(output_dir / "found_rows.csv"),
        verbose=True,
    )
    elapsed = time.perf_counter() - started_wall
    process_cpu = cpu_seconds() - started_cpu
    engine_metrics = json.loads(metrics_path.read_text())

    payload = {
        "implementation": "suql_baseline",
        "mode": "llm",
        "model": args.model,
        "cheap_model": "",
        "expensive_model": args.model,
        "cpu_seconds": process_cpu,
        "engine_seconds": float(engine_metrics["engine_seconds"]),
        "wall_seconds": elapsed,
        "llm_calls": int(engine_metrics["llm_prompts_issued"]),
        "block_join_calls": 0,
        "cheap_calls": 0,
        "expensive_calls": int(engine_metrics["llm_prompts_issued"]),
        "final_answer_rows": int(len(results)),
        "found_movie_ids": sorted(results["movie_id"].astype(str).unique()),
        "structured_candidates": int(engine_metrics["structured_candidates"]),
    }
    (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
