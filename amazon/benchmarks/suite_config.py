#!/usr/bin/env python3
"""Suite configuration helpers, extracted from the retired generalized_pipeline.

`build_local_catalog.py` needs only these two functions -- the rest of that
pipeline was replaced by the ported IMDb approaches under amazon/approaches/.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


OPS = {"eq", "ne", "lt", "le", "gt", "ge", "contains", "in"}


def apply_predicates(frame: pd.DataFrame, predicates: list[dict[str, Any]]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for item in predicates:
        col, op, value = item["column"], item["op"], item["value"]
        if col not in frame or op not in OPS:
            raise ValueError(f"Invalid structured predicate: {item}")
        series = frame[col].fillna("")
        if op in {"lt", "le", "gt", "ge"}:
            lhs, rhs = pd.to_numeric(series, errors="coerce"), float(value)
            mask &= {"lt": lhs < rhs, "le": lhs <= rhs, "gt": lhs > rhs, "ge": lhs >= rhs}[op]
        elif op == "eq":
            mask &= series.astype(str).str.casefold() == str(value).casefold()
        elif op == "ne":
            mask &= series.astype(str).str.casefold() != str(value).casefold()
        elif op == "contains":
            mask &= series.astype(str).str.contains(re.escape(str(value)), case=False, na=False)
        else:
            values = {str(v).casefold() for v in value}
            mask &= series.astype(str).str.casefold().isin(values)
    return mask


def load_config(path: Path) -> dict[str, Any]:
    cfg = read_json(path)
    required = ("structured_csv", "text_csv", "id_column", "text_column", "output_dir")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    cfg.setdefault("dataset_name", Path(cfg["structured_csv"]).stem)
    cfg.setdefault("api_base", "http://127.0.0.1:11434")
    cfg.setdefault("cheap_model", "gemma4:e2b")
    cfg.setdefault("expensive_model", "gemma4:26b")
    cfg.setdefault("minimum_ground_truth", 10)
    cfg.setdefault("suite_sizes", [5, 10])
    cfg.setdefault("candidate_pool_size", 100)
    cfg.setdefault("structured_candidate_pool_size", 50)
    cfg.setdefault("max_structured_label_candidates", 500)
    cfg.setdefault("seed", 42)
    cfg.setdefault("repetitions", 1)
    cfg.setdefault("semantic_dictionary_examples", 4)
    cfg.setdefault("cascade_target", 0.9)
    cfg.setdefault("calibration_budget", 20)
    cfg.setdefault("token_threshold", 4000)
    cfg["api_base"] = os.environ.get("GENERALIZED_API_BASE", cfg["api_base"])
    cfg["cheap_model"] = os.environ.get("GENERALIZED_CHEAP_MODEL", cfg["cheap_model"])
    cfg["expensive_model"] = os.environ.get("GENERALIZED_EXPENSIVE_MODEL", cfg["expensive_model"])
    if os.environ.get("GENERALIZED_SUITE_SIZES"):
        cfg["suite_sizes"] = [
            int(value) for value in os.environ["GENERALIZED_SUITE_SIZES"].split(",") if value.strip()
        ]
    if os.environ.get("GENERALIZED_REPETITIONS"):
        cfg["repetitions"] = int(os.environ["GENERALIZED_REPETITIONS"])
    # "none" restores the fully deterministic calibration pick, which is what
    # every run before 2026-07-27 used.
    if os.environ.get("GENERALIZED_CASCADE_SEED"):
        raw = os.environ["GENERALIZED_CASCADE_SEED"].strip()
        cfg["cascade_seed"] = None if raw.lower() == "none" else int(raw)
    cfg["_config_path"] = str(path.resolve())
    cfg["_work"] = str(Path(cfg["output_dir"]).resolve())
    # Suites and the semantic dictionary are shared, read-only inputs, but run
    # results are per-run: concurrent jobs on the same suite would otherwise
    # overwrite each other under work/outputs/<suite>.
    outputs_root = os.environ.get("GENERALIZED_OUTPUTS_ROOT")
    cfg["_outputs"] = str(
        Path(outputs_root).resolve() if outputs_root else Path(cfg["_work"]) / "outputs"
    )
    # Canonical tables, suites and the dictionary default to the work directory
    # but may be relocated independently, so a dataset can lay its benchmark
    # tree out however it likes without this generic pipeline knowing about it.
    # Relative paths resolve against the working directory, matching output_dir.
    def _rooted(key: str, default: str) -> str:
        supplied = cfg.get(key)
        return str(Path(supplied).resolve() if supplied else Path(cfg["_work"]) / default)

    cfg["_canonical"] = _rooted("canonical_dir", "canonical")
    cfg["_suites"] = _rooted("suites_dir", "suites")
    cfg["_dictionary"] = _rooted("dictionary_dir", "semantic_dictionary")
    return cfg
