from __future__ import annotations

import csv
import json
import resource
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parent


def benchmark() -> dict:
    return json.loads((ROOT / "benchmark.json").read_text())


def model_slug(model: str) -> str:
    return model.removeprefix("ollama/").removesuffix(":latest").replace(":", "_").replace("/", "_")


def pair_slug(cheap_model: str, expensive_model: str) -> str:
    return f"cheap_{model_slug(cheap_model)}__expensive_{model_slug(expensive_model)}"


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def load_movies() -> list[dict[str, str]]:
    rows = []
    with (ROOT / "data" / "imdb_structured_joined.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["text"] = (
                f"movie_id={row.get('movie_id', '')}; title={row.get('title', '')}; "
                f"year={row.get('year', '')}; director={row.get('director', '')}; "
                f"runtime={row.get('runtime', '')}; genres={row.get('genres', '')}"
            )
            rows.append(row)
    return rows


def load_reviews() -> list[dict[str, str]]:
    rows = []
    with (ROOT / "data" / "imdb_reviews.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            review = " ".join((row.get("review") or "").replace("<br />", " ").split())
            row["review"] = review
            row["text"] = f"tconst={row.get('tconst', '')}; review={review[:1800]}"
            rows.append(row)
    return rows


def truth_rows(movies: list[dict[str, str]]) -> list[dict[str, str]]:
    truth = set(benchmark()["ground_truth_movie_ids"])
    return [
        {key: movie.get(key, "") for key in ("movie_id", "title", "year", "runtime", "director", "genres")}
        for movie in movies
        if movie.get("movie_id") in truth
    ]


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

