#!/usr/bin/env python3
"""Refresh 10q human-readable files without changing held-out smaller suites."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "10q"
DATA_ROOT = ROOT.parent / "data" / "subdatasets"
SELECTIONS = {
    "10q": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    catalog = source_manifest["questions"]
    if len(catalog) != 10:
        raise RuntimeError(f"Expected 10 source questions, found {len(catalog)}")

    # Refresh the two human-readable catalog files after the 100-row catalog is built.
    source_questions: list[str] = []
    source_truth: list[str] = []
    for index, item in enumerate(catalog, 1):
        qdir = SOURCE / "per_question" / item["directory"]
        spec = json.loads((qdir / "benchmark.json").read_text())
        movies = read_csv(DATA_ROOT / "10q" / item["directory"] / "imdb_structured_joined.csv")
        by_id = {row["movie_id"]: row for row in movies}
        source_questions.extend([
            f"Q{index}: {spec['question']}", f"Semantic task: {spec['semantic_task']}",
            f"Structured filters: {json.dumps(spec['structured_filters'])}", "",
        ])
        source_truth.append(f"Q{index}: {spec['question']}")
        for movie_id in spec["ground_truth_movie_ids"]:
            row = by_id[movie_id]
            source_truth.append(f"- {row['title']} ({row['year']})")
        source_truth.append("")
    (SOURCE / "questions.txt").write_text("\n".join(source_questions).rstrip() + "\n")
    (SOURCE / "ground_truth_movies.txt").write_text("\n".join(source_truth).rstrip() + "\n")

    # The smaller suites are historical held-out prompts and must not be
    # regenerated from 10q; doing so would leak the evaluation questions.
    for suite_name, indices in SELECTIONS.items():
        if suite_name == "10q":
            continue
        suite = ROOT / suite_name
        per_question = suite / "per_question"
        if per_question.exists():
            shutil.rmtree(per_question)
        per_question.mkdir(parents=True)
        manifest = {"suite": suite_name, "question_count": len(indices), "questions": []}
        question_lines: list[str] = []
        truth_lines: list[str] = []

        for local_index, catalog_index in enumerate(indices, 1):
            source_item = catalog[catalog_index - 1]
            source_dir = SOURCE / "per_question" / source_item["directory"]
            source_data_dir = DATA_ROOT / "10q" / source_item["directory"]
            target_name = f"q_{local_index:02d}"
            target_dir = per_question / target_name
            target_dir.mkdir(parents=True)
            shutil.copy2(source_dir / "benchmark.json", target_dir / "benchmark.json")
            target_data_dir = DATA_ROOT / suite_name / target_name
            if target_data_dir.exists():
                shutil.rmtree(target_data_dir)
            shutil.copytree(source_data_dir, target_data_dir)
            spec_path = target_dir / "benchmark.json"
            spec = json.loads(spec_path.read_text())
            spec["suite_question_id"] = target_name
            spec["catalog_question_id"] = spec.get("catalog_question_id", source_item["directory"])
            spec_path.write_text(json.dumps(spec, indent=2) + "\n")
            truth_ids = set(spec["ground_truth_movie_ids"])
            if len(truth_ids) < 10:
                raise RuntimeError(f"{suite_name}/{target_name} has only {len(truth_ids)} truths")
            movies = read_csv(target_data_dir / "imdb_structured_joined.csv")
            by_id = {row["movie_id"]: row for row in movies}
            missing = truth_ids - set(by_id)
            if missing:
                raise RuntimeError(f"Missing ground-truth movie rows: {sorted(missing)}")

            manifest["questions"].append({
                "id": target_name,
                "directory": target_name,
                "catalog_directory": source_item["directory"],
                "semantic_task": spec["semantic_task"],
                "structured_filters": spec["structured_filters"],
                "ground_truth_count": len(truth_ids),
            })
            question_lines.extend([
                f"Q{local_index}: {spec['question']}",
                f"Semantic task: {spec['semantic_task']}",
                f"Structured filters: {json.dumps(spec['structured_filters'])}",
                "",
            ])
            truth_lines.append(f"Q{local_index}: {spec['question']}")
            for movie_id in spec["ground_truth_movie_ids"]:
                row = by_id[movie_id]
                truth_lines.append(f"- {row['title']} ({row['year']})")
            truth_lines.append("")

        (suite / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (suite / "questions.txt").write_text("\n".join(question_lines).rstrip() + "\n")
        (suite / "ground_truth_movies.txt").write_text("\n".join(truth_lines).rstrip() + "\n")
        (suite / "outputs").mkdir(exist_ok=True)
        (suite / "outputs" / ".gitkeep").touch()
        print(f"Built {suite_name}: {len(indices)} questions")


if __name__ == "__main__":
    main()
