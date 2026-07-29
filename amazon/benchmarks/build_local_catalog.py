#!/usr/bin/env python3
"""Build the Amazon Fashion benchmark suite and semantic dictionary once,
locally and deterministically -- no LLM, no GPU job.

Mirrors benchmarks/build_catalog.py's approach for IMDb: ground truth comes
from hand-authored regex patterns over review text (not an LLM judge), and
the semantic dictionary is mined the same deterministic way instead of
semantic_dict/mine_semantic_dict.py's embedding-based approach, since the
regex already gives a direct positive/negative label per product.

the ported approaches under amazon/approaches/
only ever read the files this script writes (work/suites/<suite>/..., work/
semantic_dictionary/semantic_dictionary.json) -- it never needs to know how
they were produced, so this script requires no pipeline.py changes.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent          # data_amazon/benchmarks
DATASET_ROOT = HERE.parent                       # data_amazon
LAB_ROOT = DATASET_ROOT.parent                   # repository root
sys.path.insert(0, str(HERE))
import suite_config as p  # noqa: E402

CONFIG_PATH = DATASET_ROOT / "config.json"
# AMAZON_QUESTIONS selects a different reviewed question set (e.g. the held-out
# ten in questions_heldout_10q.json); AMAZON_SUITE_ROOT sends the resulting
# suites and dictionary somewhere other than the config's work directory, so a
# second question set never overwrites the suites an existing run was built on.
QUESTIONS_PATH = Path(os.environ.get("AMAZON_QUESTIONS") or DATASET_ROOT / "questions_10q.json")

CANDIDATE_POOL_SIZE = 100
STRUCTURED_POOL_SIZE = 50
GROUND_TRUTH_TARGET = 15
DISTRACTOR_SPLIT = 25  # each of the two non-structured buckets
SUITE_SIZES = (1, 3, 5, 10)
DICTIONARY_EXAMPLES = 4
SCAN_LIMIT_STRUCTURED = 4000
SCAN_LIMIT_DISTRACTOR = 20000
DICTIONARY_SCAN_LIMIT = 20000

# One `|`-joined disjunction of phrase variants per concept, same style as
# benchmarks/build_catalog.py's semantic_patterns. Known limitation shared
# with that script: no negation handling (e.g. "not comfortable" still
# matches \bcomfortable\b) -- this mirrors the paper's own stated "synthetic
# lexical ground truth" limitation rather than trying to solve it here.
SEMANTIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "comfortable_dress": (r"\bcomfortable\b", r"\bcomfy\b", r"\bcomfortably\b"),
    "true_to_size_shoes": (
        r"\btrue to size\b", r"\bruns? true to size\b", r"\bfits? true to size\b",
        r"\baccurate(?:ly)? sized\b",
    ),
    "soft_fabric_shirt": (
        r"\bsoft(?:ness)?\b", r"\bsilky\b", r"\bbutter[- ]soft\b",
        r"\bcomfortable (?:fabric|material)\b", r"\bfeels? (?:so |really )?soft\b",
    ),
    "warm_jacket": (
        r"\bwarm\b", r"\bwarmth\b", r"\bkeeps? (?:me |you )?warm\b", r"\bcozy\b", r"\btoasty\b",
    ),
    "sun_protective_hat": (
        r"\bsun protection\b", r"\bUPF\b", r"\bSPF\b", r"\bsunscreen\b",
        r"\bUV protection\b", r"\bblocks? (?:harmful )?(?:the sun|UV)\b",
        r"\bprotects? .*sun\b", r"\bsunburn\b", r"\bshades? (?:you |me )?from the sun\b",
    ),
    "stylish_watch": (
        r"\bstylish\b", r"\bfashionable\b", r"\bgood[- ]looking\b",
        r"\battractive\b", r"\belegant\b", r"\bclassy\b",
    ),
    "roomy_bag": (
        r"\broomy\b", r"\bspacious\b", r"\bplenty of (?:room|space)\b",
        r"\bholds? a lot\b", r"\blots? of (?:room|space)\b",
    ),
    "durable_socks": (
        r"\bdurable\b", r"\bhold(?:s|ing)? up (?:well|great|nicely)\b",
        r"\blong[- ]lasting\b", r"\bwears? well\b",
    ),
    "walking_comfort_sandals": (r"\bcomfortable\b", r"\bcomfy\b"),
    "well_made_costume": (
        r"\bwell[- ]made\b", r"\bgood quality\b", r"\bwell (?:constructed|built)\b",
        r"\bhigh quality\b", r"\bsturdy\b",
    ),
    # --- held-out ten (questions_heldout_10q.json) -------------------------
    # Disjoint product categories and semantic axes from the ten above, so the
    # two sets can serve as independent evaluation and development suites.
    "tarnishing_necklace": (
        r"\btarnish(?:ed|es|ing)?\b",
        r"\bturn(?:ed|s)? (?:my |her |his )?(?:skin |neck |finger )?green\b",
        r"\bdiscolou?r(?:ed|s|ing|ation)?\b", r"\brust(?:ed|s|ing|y)?\b",
        r"\b(?:finish|plating|paint|coating) (?:came|wore|chipped|peeled) off\b",
        r"\bchipp(?:ed|ing)\b", r"\bpeel(?:ed|ing)\b",
    ),
    "adjustable_bracelet": (
        r"\badjustable\b", r"\badjusts?\b", r"\bresiz(?:e|ed|able)\b", r"\bextender\b",
        r"\bfits? (?:any|most|all|various) (?:wrist|size)", r"\beasy to adjust\b",
        r"\badjust(?:ed|ing) (?:the |it )?(?:size|length|fit)\b",
    ),
    # Condition-on-arrival rather than sun/glare: the latter is barely
    # expressible outside eyewear, so it could not raise the 25 semantic-only
    # distractors the pool needs, and it overlapped sun_protective_hat above.
    "damaged_on_arrival_sunglasses": (
        r"\bscratch(?:ed|es|ing)?\b", r"\bdamaged\b", r"\bbroken\b", r"\bcracked\b",
        r"\bdefective\b", r"\barrived (?:broken|damaged|scratched|cracked)\b",
        r"\bdent(?:ed|s)?\b", r"\bbent\b",
    ),
    "itchy_sweater": (
        r"\bitch(?:y|ed|es|ing)?\b", r"\bscratchy\b", r"\birritat(?:e|es|ed|ing|ion)\b",
        r"\bprickly\b", r"\brough (?:on|against) (?:the |my )?skin\b", r"\bmakes? me itch\b",
    ),
    "poor_wash_durability_hoodie": (
        r"\bshr(?:ank|unk|inks?|inking)\b", r"\bfad(?:e|ed|es|ing)\b", r"\bpill(?:ed|ing|s)\b",
        r"\bstretched out\b", r"\bafter (?:one |a |the |first |1 )?wash",
        r"\bin the (?:wash|dryer)\b", r"\bfell apart\b", r"\bcame apart\b",
    ),
    "slim_wallet": (
        r"\bslim\b", r"\bthin\b", r"\bcompact\b",
        r"\bfits? (?:in|into) (?:my |your |the )?(?:front |back )?pocket\b",
        r"\bnot bulky\b", r"\blow profile\b", r"\bdoes ?n.t add bulk\b",
    ),
    "inaccurate_sizing_jeans": (
        r"\bruns? (?:a bit |a little |very |really )?(?:small|large|big|tight)\b",
        r"\bsiz(?:e|ed) (?:up|down)\b", r"\border(?:ed)? (?:a )?size (?:up|down)\b",
        r"\btoo (?:small|big|large|tight|loose)\b",
        r"\bsmaller than (?:expected|usual|normal)\b",
        r"\blarger than (?:expected|usual|normal)\b",
    ),
    "colour_mismatch_scarf": (
        r"\bcolou?r (?:is |was |looks? )?(?:different|not)\b",
        r"\bnot (?:the )?(?:same )?colou?r\b",
        r"\bdifferent (?:from|than) (?:the )?(?:picture|photo|image)\b",
        r"\b(?:darker|lighter|brighter|duller) than (?:the )?(?:picture|photo|expected|shown|it)\b",
        r"\bnot as (?:pictured|shown|described)\b",
        r"\bdoes ?n.t match the (?:picture|photo|description)\b",
        r"\bnothing like the (?:picture|photo)\b",
    ),
    "supportive_swimsuit": (
        r"\bsupport(?:ive|s)?\b", r"\bcoverage\b", r"\bstays? (?:in )?place\b",
        r"\bsecure(?:ly)?\b", r"\bcovers? (?:well|everything)\b",
        r"\bhold(?:s)? (?:me |you |everything )?in\b",
    ),
    "sheer_leggings": (
        r"\bsheer\b", r"\bsee[- ]?thr(?:ough|u)\b", r"\btransparent\b", r"\btoo thin\b",
        r"\bcan see (?:my |your |right )?thr(?:ough|u)\b", r"\bnot opaque\b",
    ),
}


def semantic_match(text: str, concept: str) -> "re.Match[str] | None":
    pattern = "(?:" + "|".join(SEMANTIC_PATTERNS[concept]) + ")"
    return re.search(pattern, text, flags=re.IGNORECASE)


def combined_text_map(texts: pd.DataFrame, ident: str, text_col: str, ids: list[str]) -> dict[str, str]:
    subset = texts[texts[ident].astype(str).isin(set(ids))]
    grouped = subset.groupby(ident)[text_col].apply(lambda s: "\n\n".join(s)[:4000])
    return grouped.to_dict()


def label_pool(ids: list[str], text_map: dict[str, str], concept: str) -> tuple[list[dict], list[dict]]:
    positives, negatives = [], []
    for pid in ids:
        text = text_map.get(pid, "")
        if not text:
            continue
        match = semantic_match(text, concept)
        if match:
            positives.append({"id": pid, "text": text, "evidence": match.group(0)})
        else:
            negatives.append({"id": pid, "text": text})
    return positives, negatives


def build_question(
    cfg: dict[str, Any], entity_eval: pd.DataFrame, texts: pd.DataFrame,
    ident: str, text_col: str, spec: dict[str, Any], rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    concept = spec["semantic_concept"]
    mask = p.apply_predicates(entity_eval, spec["structured_predicates"])
    structured_ids = entity_eval.loc[mask, ident].tolist()
    other_ids = entity_eval.loc[~mask, ident].tolist()
    rng.shuffle(structured_ids)
    rng.shuffle(other_ids)
    structured_scan = structured_ids[:SCAN_LIMIT_STRUCTURED]
    other_scan = other_ids[:SCAN_LIMIT_DISTRACTOR]

    structured_text = combined_text_map(texts, ident, text_col, structured_scan)
    other_text = combined_text_map(texts, ident, text_col, other_scan)

    structured_pos, structured_neg = label_pool(structured_scan, structured_text, concept)
    other_pos, other_neg = label_pool(other_scan, other_text, concept)

    minimum_truth = int(cfg["minimum_ground_truth"])
    if len(structured_pos) < minimum_truth:
        raise RuntimeError(
            f"{concept!r}: only {len(structured_pos)} structured&semantic matches "
            f"(scanned {len(structured_scan)} of {len(structured_ids)} structured candidates); "
            f"need >= {minimum_truth}. Loosen SEMANTIC_PATTERNS[{concept!r}] or raise the scan limit."
        )

    ground_truth = structured_pos[: max(minimum_truth, GROUND_TRUTH_TARGET)]
    remaining_structured = STRUCTURED_POOL_SIZE - len(ground_truth)
    structured_negatives = structured_neg[:remaining_structured]
    if len(structured_negatives) < remaining_structured:
        raise RuntimeError(
            f"{concept!r}: only {len(structured_negatives)} structured&~semantic distractors "
            f"available, need {remaining_structured}."
        )

    other_pos_take = other_pos[:DISTRACTOR_SPLIT]
    other_neg_take = other_neg[:DISTRACTOR_SPLIT]
    if len(other_pos_take) < DISTRACTOR_SPLIT or len(other_neg_take) < DISTRACTOR_SPLIT:
        raise RuntimeError(
            f"{concept!r}: distractor pool too small "
            f"(semantic-only={len(other_pos_take)}, double-negative={len(other_neg_take)}, need {DISTRACTOR_SPLIT} each)."
        )

    candidate_records = ground_truth + structured_negatives + other_pos_take + other_neg_take
    assert len(candidate_records) == CANDIDATE_POOL_SIZE, len(candidate_records)
    candidate_ids = [record["id"] for record in candidate_records]
    ground_truth_ids = sorted(record["id"] for record in ground_truth)

    benchmark_common = {
        "question": spec["question"],
        "semantic_question": spec["semantic_question"],
        "semantic_concept": concept,
        "structured_predicates": spec["structured_predicates"],
        "candidate_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "ground_truth_ids": ground_truth_ids,
        "ground_truth_count": len(ground_truth_ids),
    }
    return benchmark_common, candidate_records


def build_dictionary(
    cfg: dict[str, Any], entity: pd.DataFrame, texts: pd.DataFrame,
    ident: str, text_col: str, question_specs: list[dict[str, Any]], rng: random.Random,
) -> dict[str, Any]:
    dictionary_ids = entity.loc[entity["_split"] == "dictionary", ident].tolist()
    rng.shuffle(dictionary_ids)
    scan_ids = dictionary_ids[:DICTIONARY_SCAN_LIMIT]
    text_map = combined_text_map(texts, ident, text_col, scan_ids)

    concepts: dict[str, Any] = {}
    for spec in question_specs:
        concept = spec["semantic_concept"]
        positives, negatives = [], []
        for pid in scan_ids:
            text = text_map.get(pid, "")
            if not text:
                continue
            match = semantic_match(text, concept)
            if match and len(positives) < DICTIONARY_EXAMPLES:
                positives.append({"id": pid, "text": text[:2000], "evidence": match.group(0)})
            elif not match and len(negatives) < DICTIONARY_EXAMPLES:
                negatives.append({"id": pid, "text": text[:2000], "evidence": ""})
            if len(positives) >= DICTIONARY_EXAMPLES and len(negatives) >= DICTIONARY_EXAMPLES:
                break
        concepts[concept] = {
            "semantic_question": spec["semantic_question"],
            "positive_examples": positives,
            "negative_examples": negatives,
        }
    return {
        "provenance": {
            "split": "dictionary", "seed": cfg["seed"],
            "method": "deterministic_regex_local (mirrors benchmarks/build_catalog.py's approach for IMDb)",
        },
        "concepts": concepts,
    }


def main() -> None:
    cfg = p.load_config(CONFIG_PATH)
    canonical = Path(cfg["_canonical"])
    # Canonical tables are always read from the configured canonical directory;
    # only the generated suites and dictionary follow AMAZON_SUITE_ROOT.
    out_root = Path(os.environ.get("AMAZON_SUITE_ROOT") or cfg["_suites"])
    entity = pd.read_csv(canonical / "structured.csv", dtype=str).fillna("")
    texts = pd.read_csv(canonical / "texts.csv", dtype=str).fillna("")
    ident, text_col = cfg["id_column"], cfg["text_column"]
    entity_eval = entity[entity["_split"] == "evaluation"].reset_index(drop=True)

    question_specs = json.loads(QUESTIONS_PATH.read_text())["questions"]
    rng = random.Random(int(cfg["seed"]))

    built: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for spec in question_specs:
        benchmark_common, candidate_records = build_question(
            cfg, entity_eval, texts, ident, text_col, spec, rng
        )
        built.append((spec, benchmark_common, candidate_records))
        print(
            f"{benchmark_common['semantic_concept']}: "
            f"candidates={benchmark_common['candidate_count']} "
            f"ground_truth={benchmark_common['ground_truth_count']}"
        )

    for suite_size in SUITE_SIZES:
        root = out_root / f"{suite_size}q"
        root.mkdir(parents=True, exist_ok=True)
        manifest = {"suite": f"{suite_size}q", "question_count": suite_size, "questions": []}
        question_lines: list[str] = []
        truth_lines: list[str] = []
        for index, (spec, benchmark_common, candidate_records) in enumerate(built[:suite_size], 1):
            qid = f"q_{index:02d}"
            qroot = root / "per_question" / qid
            qroot.mkdir(parents=True, exist_ok=True)
            benchmark = {**benchmark_common, "id": qid}
            (qroot / "benchmark.json").write_text(
                json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            manifest["questions"].append({
                "id": qid, "directory": qid, "semantic_concept": spec["semantic_concept"],
                "ground_truth_count": benchmark_common["ground_truth_count"],
            })
            question_lines.append(f"{qid}: {spec['question']}")
            truth_lines.append(f"{qid}: {spec['question']}")
            truth_by_id = {r["id"]: r for r in candidate_records if r["id"] in set(benchmark_common["ground_truth_ids"])}
            for gid in benchmark_common["ground_truth_ids"]:
                evidence = truth_by_id[gid].get("evidence", "")
                truth_lines.append(f"  - {gid}: {evidence!r}")
            truth_lines.append("")
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (root / "questions.txt").write_text("\n".join(question_lines) + "\n", encoding="utf-8")
        (root / "ground_truth_summary.txt").write_text("\n".join(truth_lines).rstrip() + "\n", encoding="utf-8")

    dictionary = build_dictionary(cfg, entity, texts, ident, text_col, question_specs, rng)
    dict_path = (
        Path(os.environ["AMAZON_SUITE_ROOT"]) / "semantic_dict"
        if os.environ.get("AMAZON_SUITE_ROOT")
        else Path(cfg["_dictionary"])
    ) / "semantic_dictionary.json"
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    dict_path.write_text(json.dumps(dictionary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for concept, entry in dictionary["concepts"].items():
        print(
            f"dictionary[{concept}]: "
            f"positives={len(entry['positive_examples'])} negatives={len(entry['negative_examples'])}"
        )
    print("done")


if __name__ == "__main__":
    main()
