#!/usr/bin/env python3
"""Materialise Amazon suites as per-question CSVs in the IMDb runner's layout.

The four Amazon approaches are a port of the IMDb ones, so they expect the same
per-question files the IMDb runner reads:

    amazon/data/subdatasets/<suite>/q_XX/
        amazon_structured_joined.csv   product_id,title,brand,category,price
        amazon_reviews.csv             product_id,review
        amazon_joined.csv              both, joined on product_id
        ground_truth.csv               product_id,title

The suites themselves (`benchmarks/<suite>/per_question/q_XX/benchmark.json`)
list candidate and ground-truth ids; this script resolves those ids against the
canonical tables and writes the CSVs. It reads only -- suites are not rebuilt.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent          # amazon/benchmarks
DATASET_ROOT = HERE.parent                       # amazon
CANONICAL = DATASET_ROOT / "canonical"
OUT_ROOT = DATASET_ROOT / "data" / "subdatasets"

STRUCTURED_COLUMNS = ["product_id", "title", "brand", "category", "price"]
SUITES = ("1q", "3q", "5q", "10q")


def load_canonical() -> tuple[pd.DataFrame, dict[str, str]]:
    entity = pd.read_csv(CANONICAL / "structured.csv", dtype=str).fillna("")
    texts = pd.read_csv(CANONICAL / "texts.csv", dtype=str).fillna("")
    # One combined review per product, matching the IMDb suites' one-review-per-row shape.
    by_id = (
        texts.groupby("product_id")["review_text"]
        .apply(lambda s: "\n\n".join(s)[:4000])
        .to_dict()
    )
    return entity, by_id


def write_question(out_dir: Path, rows: list[dict], truth_ids: set[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, fieldnames: list[str], records: list[dict]) -> None:
        with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({k: record.get(k, "") for k in fieldnames})

    dump("amazon_structured_joined.csv", STRUCTURED_COLUMNS, rows)
    dump("amazon_reviews.csv", ["product_id", "review"], rows)
    dump("amazon_joined.csv", STRUCTURED_COLUMNS + ["review"], rows)
    dump(
        "ground_truth.csv",
        ["product_id", "title"],
        [r for r in rows if r["product_id"] in truth_ids],
    )


def main() -> None:
    entity, reviews_by_id = load_canonical()
    entity_by_id = {str(r["product_id"]): r for r in entity.to_dict("records")}

    for suite in SUITES:
        suite_root = HERE / suite
        manifest_path = suite_root / "manifest.json"
        if not manifest_path.exists():
            print(f"{suite}: no manifest, skipped")
            continue
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["questions"]:
            qid = item["directory"]
            spec = json.loads(
                (suite_root / "per_question" / qid / "benchmark.json").read_text()
            )
            truth_ids = {str(v) for v in spec["ground_truth_ids"]}
            rows = []
            for pid in (str(v) for v in spec["candidate_ids"]):
                base = entity_by_id.get(pid)
                if base is None:
                    continue
                row = {c: base.get(c, "") for c in STRUCTURED_COLUMNS}
                row["review"] = reviews_by_id.get(pid, "")
                rows.append(row)
            missing_truth = truth_ids - {r["product_id"] for r in rows}
            if missing_truth:
                raise SystemExit(
                    f"{suite}/{qid}: {len(missing_truth)} ground-truth ids missing "
                    f"from the candidate pool -- suite and canonical tables disagree"
                )
            write_question(OUT_ROOT / suite / qid, rows, truth_ids)
        print(f"{suite}: wrote {len(manifest['questions'])} question directories")


if __name__ == "__main__":
    main()
