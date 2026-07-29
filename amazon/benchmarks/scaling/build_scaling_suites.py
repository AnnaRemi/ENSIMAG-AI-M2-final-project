#!/usr/bin/env python3
"""Build one question's candidate pool at several sizes N, each as its own
1-question suite, to study how cost/calls/time scale with N per method.

This backs a specific claim about cascading: baseline (suql_baseline) cost is
O(N) with zero fixed overhead. A cascade (trummer_v1) is also O(N), but pays
a *fixed* calibration cost (calibration_budget expensive calls, in addition
to escalating hard cases) regardless of N. For small N that fixed cost can
make the cascade *more* expensive than baseline; as N grows, the fixed cost
is amortized over more candidates and -- provided the learned threshold lets
a meaningful fraction of items skip escalation -- the cascade should
eventually cross below baseline. This script builds the suites needed to
find that crossover; run_scaling_experiment.py runs them and plots it.

Reuses build_local_catalog.py's semantic-matching machinery (same regex
patterns, same deterministic split boundaries) -- just parameterizes the
pool sizes instead of using its fixed CANDIDATE_POOL_SIZE/STRUCTURED_POOL_SIZE.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent.parent.parent  # data_amazon/benchmarks/scaling -> repo root
sys.path.insert(0, str(LAB_ROOT / "generalized_pipeline"))
import pipeline as p  # noqa: E402
sys.path.insert(0, str(HERE.parent))
import build_local_catalog as catalog  # noqa: E402

CONCEPT = "comfortable_dress"
ROW_COUNTS = (25, 50, 100, 200, 400, 800)
SUITE_PREFIX = "scale_n"


def suite_name(n: int) -> str:
    return f"{SUITE_PREFIX}{n:04d}"


def build_question_at_scale(
    entity_eval: pd.DataFrame, texts: pd.DataFrame, ident: str, text_col: str,
    spec: dict[str, Any], rng: random.Random, n: int,
) -> dict[str, Any]:
    concept = spec["semantic_concept"]
    mask = p.apply_predicates(entity_eval, spec["structured_predicates"])
    structured_ids = entity_eval.loc[mask, ident].tolist()
    other_ids = entity_eval.loc[~mask, ident].tolist()
    rng.shuffle(structured_ids)
    rng.shuffle(other_ids)

    structured_pool_size = max(2, n // 2)
    distractor_pool_size = n - structured_pool_size
    ground_truth_target = max(2, round(0.3 * structured_pool_size))

    scan_limit_structured = max(catalog.SCAN_LIMIT_STRUCTURED, structured_pool_size * 4)
    scan_limit_distractor = max(catalog.SCAN_LIMIT_DISTRACTOR, distractor_pool_size * 4)
    structured_scan = structured_ids[:scan_limit_structured]
    other_scan = other_ids[:scan_limit_distractor]

    structured_text = catalog.combined_text_map(texts, ident, text_col, structured_scan)
    other_text = catalog.combined_text_map(texts, ident, text_col, other_scan)

    structured_pos, structured_neg = catalog.label_pool(structured_scan, structured_text, concept)
    other_pos, other_neg = catalog.label_pool(other_scan, other_text, concept)

    if len(structured_pos) < ground_truth_target:
        raise RuntimeError(f"n={n}: only {len(structured_pos)} structured&semantic matches, need {ground_truth_target}")
    ground_truth = structured_pos[:ground_truth_target]
    remaining_structured = structured_pool_size - len(ground_truth)
    structured_negatives = structured_neg[:remaining_structured]
    if len(structured_negatives) < remaining_structured:
        raise RuntimeError(
            f"n={n}: only {len(structured_negatives)} structured&~semantic distractors, need {remaining_structured}"
        )

    half = distractor_pool_size // 2
    other_pos_take = other_pos[:half]
    other_neg_take = other_neg[:distractor_pool_size - half]
    if len(other_pos_take) < half or len(other_neg_take) < (distractor_pool_size - half):
        raise RuntimeError(
            f"n={n}: distractor pool too small "
            f"(semantic-only={len(other_pos_take)}/{half}, double-negative={len(other_neg_take)}/{distractor_pool_size - half})"
        )

    candidate_records = ground_truth + structured_negatives + other_pos_take + other_neg_take
    assert len(candidate_records) == n, (len(candidate_records), n)
    candidate_ids = [r["id"] for r in candidate_records]
    ground_truth_ids = sorted(r["id"] for r in ground_truth)

    return {
        "question": spec["question"], "semantic_question": spec["semantic_question"],
        "semantic_concept": concept, "structured_predicates": spec["structured_predicates"],
        "candidate_ids": candidate_ids, "candidate_count": len(candidate_ids),
        "ground_truth_ids": ground_truth_ids, "ground_truth_count": len(ground_truth_ids),
    }


def build_all(row_counts: tuple[int, ...] = ROW_COUNTS, concept: str = CONCEPT) -> None:
    cfg = p.load_config(catalog.CONFIG_PATH)
    work = Path(cfg["_work"])
    entity = pd.read_csv(work / "canonical/structured.csv", dtype=str).fillna("")
    texts = pd.read_csv(work / "canonical/texts.csv", dtype=str).fillna("")
    ident, text_col = cfg["id_column"], cfg["text_column"]
    entity_eval = entity[entity["_split"] == "evaluation"].reset_index(drop=True)

    question_specs = json.loads(catalog.QUESTIONS_PATH.read_text())["questions"]
    spec = next(s for s in question_specs if s["semantic_concept"] == concept)
    rng = random.Random(int(cfg["seed"]))

    for n in row_counts:
        benchmark_common = build_question_at_scale(entity_eval, texts, ident, text_col, spec, rng, n)
        suite = suite_name(n)
        root = work / "suites" / suite
        qroot = root / "per_question" / "q_01"
        qroot.mkdir(parents=True, exist_ok=True)
        benchmark = {**benchmark_common, "id": "q_01"}
        (qroot / "benchmark.json").write_text(json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {
            "suite": suite, "question_count": 1,
            "questions": [{
                "id": "q_01", "directory": "q_01", "semantic_concept": concept,
                "ground_truth_count": benchmark_common["ground_truth_count"],
            }],
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"n={n}: candidates={benchmark_common['candidate_count']} ground_truth={benchmark_common['ground_truth_count']}")


if __name__ == "__main__":
    build_all()
