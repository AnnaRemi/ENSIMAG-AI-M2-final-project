from __future__ import annotations

import csv
import json
import resource
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parent


def question_dir(name: str) -> Path:
    path = ROOT / name
    if not (path / "benchmark.json").exists():
        raise FileNotFoundError(f"Unknown question directory: {path}")
    return path


def benchmark(name: str) -> dict:
    return json.loads((question_dir(name) / "benchmark.json").read_text())


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def load_movies(name: str) -> list[dict[str, str]]:
    rows = []
    path = question_dir(name) / "data" / "imdb_structured_joined.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["text"] = (
                f"movie_id={row.get('movie_id', '')}; title={row.get('title', '')}; "
                f"year={row.get('year', '')}; director={row.get('director', '')}; "
                f"runtime={row.get('runtime', '')}; genres={row.get('genres', '')}"
            )
            rows.append(row)
    return rows


def load_reviews(name: str) -> list[dict[str, str]]:
    rows = []
    path = question_dir(name) / "data" / "imdb_reviews.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            review = " ".join((row.get("review") or "").replace("<br />", " ").split())
            row["review"] = review
            row["text"] = f"tconst={row.get('tconst', '')}; review={review[:1800]}"
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["movie_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
