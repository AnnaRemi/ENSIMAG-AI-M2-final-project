#!/usr/bin/env python3
"""Run the row-count scaling experiment built by build_scaling_suites.py:
1 repetition per (row count N, method), collecting llm_calls/wall_seconds/
cost so the crossover between suql_baseline (vanilla) and trummer_v1
(structured filter + batched cascade) can be found and plotted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent.parent.parent  # data_amazon/benchmarks/scaling -> repo root
sys.path.insert(0, str(LAB_ROOT / "generalized_pipeline"))
import pipeline as p  # noqa: E402
sys.path.insert(0, str(HERE))
import build_scaling_suites as scaling  # noqa: E402


def run_all(
    row_counts: list[int], methods: list[str], cheap_model: str | None, expensive_model: str | None,
    repetitions: int = 1,
) -> pd.DataFrame:
    cfg = p.load_config(scaling.catalog.CONFIG_PATH)
    cfg["repetitions"] = repetitions
    if cheap_model:
        cfg["cheap_model"] = cheap_model
    if expensive_model:
        cfg["expensive_model"] = expensive_model

    rows = []
    for n in row_counts:
        suite = scaling.suite_name(n)
        manifest_path = Path(cfg["_work"]) / "suites" / suite / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} missing -- run build_scaling_suites.py first")
        # calibration_budget fixed regardless of N gives the cascade a shrinking,
        # less-representative calibration sample as N grows (e.g. 20/50=40% of
        # candidates at n=100 vs 20/150=13% at n=300), which degrades the
        # learned routing threshold. Scale it with N instead.
        cfg["calibration_budget"] = max(1, n // 5)
        print(f"=== n={n} ({suite}) calibration_budget={cfg['calibration_budget']} ===")
        p.run_suite(cfg, suite, methods)
        comparison = pd.read_csv(Path(cfg["_work"]) / "outputs" / suite / "comparison.csv")
        comparison.insert(0, "n", n)
        rows.append(comparison)

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-counts", type=int, nargs="+", default=list(scaling.ROW_COUNTS))
    ap.add_argument(
        "--methods", nargs="+",
        default=["suql_baseline", "suql_v1", "trummer_baseline", "trummer_v1"],
    )
    ap.add_argument("--cheap-model", default=None)
    ap.add_argument("--expensive-model", default=None)
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--output", type=Path, default=HERE / "scaling_experiment.csv")
    args = ap.parse_args()

    result = run_all(args.row_counts, args.methods, args.cheap_model, args.expensive_model, args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")
    numeric = ["precision", "recall", "f1", "wall_seconds", "llm_calls", "cheap_calls", "expensive_calls"]
    summary = result.groupby(["n", "method"], as_index=False)[numeric].mean()
    print(summary.sort_values(["n", "method"]).to_string(index=False))


if __name__ == "__main__":
    main()
