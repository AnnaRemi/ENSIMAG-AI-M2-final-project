#!/usr/bin/env python3
"""Mine prompt context for semantic review predicates.

Anchors and examples produced here are prompt hints for an LLM judge. They are
not, and must not be used as, an inference-time keyword classifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from semantic_dict_loader import validate_semantic_dict


TOKEN_RE = re.compile(r"(?u)\b[\w][\w'-]*\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
VALID_TEMPLATE_TYPES = {"recommend", "praise", "describe", "criticize"}
SAMPLE_SIZE = 10_000
RANDOM_SEED = 42
MINING_FRACTION = 0.70
LOW_SAMPLE_CUTOFF = 50


@dataclass(frozen=True)
class Review:
    review_id: str
    movie_id: str
    text: str


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="CSV, JSONL, JSON, or Parquet review corpus")
    parser.add_argument("--categories", type=Path, default=here / "categories.json")
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--movie-id-column", default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--review-id-column", default=None)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--embedding-backend", choices=("sentence-transformers", "hashing"),
        default="sentence-transformers",
        help="Hashing exists for offline smoke tests; production mining should use SentenceTransformers.",
    )
    parser.add_argument("--min-ngram-document-frequency", type=int, default=2)
    return parser.parse_args()


def _pick_column(names: Sequence[str], explicit: str | None, aliases: Sequence[str], kind: str) -> str:
    if explicit:
        if explicit not in names:
            raise ValueError(f"{kind} column {explicit!r} is absent; columns={list(names)}")
        return explicit
    lowered = {name.lower(): name for name in names}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    raise ValueError(f"Could not infer {kind} column from {list(names)}; pass it explicitly")


def _records_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError("JSON dataset must be an array of review objects")
        return value
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas and a Parquet engine are required for Parquet input") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_reviews(
    path: Path,
    movie_id_column: str | None = None,
    text_column: str | None = None,
    review_id_column: str | None = None,
) -> list[Review]:
    records = _records_from_file(path)
    if not records:
        raise ValueError("Dataset is empty")
    names = list(records[0].keys())
    movie_col = _pick_column(names, movie_id_column, ("movie_id", "tconst", "imdb_id"), "movie id")
    text_col = _pick_column(names, text_column, ("review_text", "review", "text"), "review text")
    review_col = None
    if review_id_column:
        review_col = _pick_column(names, review_id_column, (), "review id")
    else:
        for alias in ("review_id", "id"):
            if alias in {name.lower() for name in names}:
                review_col = next(name for name in names if name.lower() == alias)
                break
    unique: dict[tuple[str, str], Review] = {}
    for record in records:
        movie_id = str(record.get(movie_col, "")).strip()
        text = html.unescape(HTML_TAG_RE.sub(" ", str(record.get(text_col, ""))))
        text = " ".join(text.split())
        if not movie_id or not text:
            continue
        key = (movie_id, text)
        if key in unique:
            continue
        supplied_id = str(record.get(review_col, "")).strip() if review_col else ""
        digest = hashlib.sha256(f"{movie_id}\0{text}".encode("utf-8")).hexdigest()[:24]
        unique[key] = Review(supplied_id or f"sha256:{digest}", movie_id, text)
    if not unique:
        raise ValueError("No non-empty reviews remain after loading")
    return list(unique.values())


def load_categories(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        categories = json.load(handle)
    if not isinstance(categories, list) or not categories:
        raise ValueError("Category schema must be a non-empty JSON array")
    required = {"category_id", "template_type", "attribute", "seed_regex", "topic_regex"}
    seen: set[str] = set()
    for index, category in enumerate(categories):
        if not isinstance(category, dict) or not required <= category.keys():
            raise ValueError(f"Category {index} must contain {sorted(required)}")
        category_id = category["category_id"]
        if not isinstance(category_id, str) or not category_id or category_id in seen:
            raise ValueError(f"Invalid or duplicate category_id: {category_id!r}")
        if category["template_type"] not in VALID_TEMPLATE_TYPES:
            raise ValueError(f"{category_id}: unsupported template_type")
        re.compile(category["seed_regex"], re.IGNORECASE)
        re.compile(category["topic_regex"], re.IGNORECASE)
        seen.add(category_id)
    return categories


def split_sample(reviews: Sequence[Review], requested: int, seed: int) -> tuple[list[Review], list[Review]]:
    rng = random.Random(seed)
    sampled = rng.sample(list(reviews), min(requested, len(reviews)))
    rng.shuffle(sampled)
    mining_size = int(math.floor(len(sampled) * MINING_FRACTION))
    return sampled[:mining_size], sampled[mining_size:]


def tokenize_ngrams(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return [" ".join(tokens[i:i + n]) for n in (1, 2, 3) for i in range(len(tokens) - n + 1)]


def mine_log_odds_anchors(
    texts: Sequence[str], labels: Sequence[bool], top_n: int = 20, min_df: int = 2
) -> list[dict[str, float | str]]:
    positive = Counter()
    negative = Counter()
    pooled = Counter()
    document_frequency = Counter()
    for text, label in zip(texts, labels):
        grams = tokenize_ngrams(text)
        counts = Counter(grams)
        (positive if label else negative).update(counts)
        pooled.update(counts)
        document_frequency.update(counts.keys())
    vocabulary = [term for term, df in document_frequency.items() if df >= min_df]
    if not vocabulary or not any(labels) or all(labels):
        return []
    pos_total = sum(positive[term] for term in vocabulary)
    neg_total = sum(negative[term] for term in vocabulary)
    pooled_total = pos_total + neg_total
    # An informative pooled-corpus prior with total mass equal to vocabulary size.
    alpha_0 = float(len(vocabulary))
    scored: list[tuple[str, float]] = []
    for term in vocabulary:
        alpha_i = alpha_0 * pooled[term] / pooled_total
        pos = positive[term]
        neg = negative[term]
        pos_other = max(pos_total + alpha_0 - pos - alpha_i, 1e-12)
        neg_other = max(neg_total + alpha_0 - neg - alpha_i, 1e-12)
        delta = math.log((pos + alpha_i) / pos_other) - math.log((neg + alpha_i) / neg_other)
        variance = 1.0 / (pos + alpha_i) + 1.0 / (neg + alpha_i)
        z_score = delta / math.sqrt(variance)
        if z_score > 0:
            scored.append((term, z_score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [{"term": term, "z_score": round(score, 6)} for term, score in scored[:top_n]]


class Embedder:
    def __init__(self, backend: str, model_name: str, seed: int):
        self.backend = backend
        self.model_name = model_name
        self.seed = seed
        self._model: Any = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install sentence-transformers or use --embedding-backend hashing only for an offline smoke test"
                ) from exc
            if self._model is None:
                self._model = SentenceTransformer(self.model_name)
            return np.asarray(
                self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
        # Deterministic signed feature hashing; deliberately opt-in and marked in provenance.
        matrix = np.zeros((len(texts), 512), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(tokenize_ngrams(text)).items():
                digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                matrix[row, value % 512] += count if value & 1 else -count
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def diverse_indices(embeddings: np.ndarray, count: int, seed: int) -> list[int]:
    n_rows = len(embeddings)
    if n_rows <= count:
        return list(range(n_rows))
    rng = np.random.default_rng(seed)
    # k-means++ initialization followed by deterministic Lloyd iterations.
    centers = [int(rng.integers(n_rows))]
    distances = np.sum((embeddings - embeddings[centers[0]]) ** 2, axis=1)
    for _ in range(1, count):
        total = float(distances.sum())
        next_index = int(rng.integers(n_rows)) if total <= 0 else int(rng.choice(n_rows, p=distances / total))
        centers.append(next_index)
        distances = np.minimum(distances, np.sum((embeddings - embeddings[next_index]) ** 2, axis=1))
    centroids = embeddings[centers].copy()
    assignments = np.full(n_rows, -1, dtype=np.int32)
    for _ in range(100):
        new_assignments = np.argmin(((embeddings[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster in range(count):
            members = embeddings[assignments == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    selected: list[int] = []
    for cluster in range(count):
        member_indices = np.flatnonzero(assignments == cluster)
        if not len(member_indices):
            continue
        nearest = member_indices[np.argmin(((embeddings[member_indices] - centroids[cluster]) ** 2).sum(axis=1))]
        selected.append(int(nearest))
    return selected


def excerpt_around_match(text: str, pattern: re.Pattern[str], limit: int = 300) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    match = pattern.search(normalized)
    center = (match.start() + match.end()) // 2 if match else len(normalized) // 2
    start = max(0, min(center - limit // 2, len(normalized) - limit))
    if start:
        boundary = normalized.find(" ", start)
        start = boundary + 1 if boundary != -1 else start
    end = min(len(normalized), start + limit)
    if end < len(normalized):
        boundary = normalized.rfind(" ", start, end)
        end = boundary if boundary > start else end
    return ("…" if start else "") + normalized[start:end].strip() + ("…" if end < len(normalized) else "")


def select_examples(
    reviews: Sequence[Review], pattern: re.Pattern[str], embedder: Embedder, seed: int, count: int = 5
) -> list[str]:
    if not reviews:
        return []
    embeddings = embedder.encode([review.text for review in reviews])
    indices = diverse_indices(embeddings, min(count, len(reviews)), seed)
    return [excerpt_around_match(reviews[index].text, pattern) for index in indices]


def persist_splits(
    path: Path, mining: Sequence[Review], holdout: Sequence[Review], source: Path, requested: int, seed: int
) -> None:
    value = {
        "source_dataset": str(source.resolve()),
        "review_id_scheme": "source review ID when supplied, otherwise sha256(movie_id + NUL + review_text)[:24]",
        "requested_sample_size": requested,
        "actual_sample_size": len(mining) + len(holdout),
        "random_seed": seed,
        "mining_review_ids": [review.review_id for review in mining],
        "holdout_review_ids": [review.review_id for review in holdout],
    }
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(path: Path, semantic_dict: dict[str, dict[str, Any]]) -> None:
    lines = ["# Semantic dictionary mining report", "", "Anchors and examples require human review before production use.", ""]
    for category_id, entry in semantic_dict.items():
        stats = entry["mining_stats"]
        lines.extend([
            f"## {category_id}", "",
            f"- Proxy positive rate: {stats['proxy_positive_rate']:.4f} ({stats['proxy_positive_count']}/{stats['mining_set_size']})",
            f"- Low-sample warning: {'yes' if stats['low_sample_warning'] else 'no'}",
            f"- Top anchors: {', '.join(anchor['term'] for anchor in entry['lexical_anchors'][:5]) or '(none)'}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(semantic_dict: dict[str, dict[str, Any]]) -> None:
    print(f"{'category':26} {'positive rate':>13} {'anchors':>8}  warning")
    print("-" * 62)
    for category_id, entry in semantic_dict.items():
        stats = entry["mining_stats"]
        print(f"{category_id:26} {stats['proxy_positive_rate']:13.4f} {len(entry['lexical_anchors']):8d}  "
              f"{'LOW SAMPLE' if stats['low_sample_warning'] else ''}")


def run(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    reviews = load_reviews(args.dataset, args.movie_id_column, args.text_column, args.review_id_column)
    categories = load_categories(args.categories)
    mining, holdout = split_sample(reviews, args.sample_size, args.seed)
    if not mining:
        raise ValueError("At least two sampled reviews are needed to create a non-empty mining split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    persist_splits(args.output_dir / "semantic_dict_splits.json", mining, holdout, args.dataset, args.sample_size, args.seed)
    print(f"Loaded {len(reviews)} unique reviews; sampled {len(mining) + len(holdout)}: "
          f"mining={len(mining)}, holdout={len(holdout)}")
    mined_at = datetime.now(timezone.utc).isoformat()
    embedder = Embedder(args.embedding_backend, args.embedding_model, args.seed)
    texts = [review.text for review in mining]
    output: dict[str, dict[str, Any]] = {}
    for category_index, category in enumerate(categories):
        seed_pattern = re.compile(category["seed_regex"], re.IGNORECASE)
        topic_pattern = re.compile(category["topic_regex"], re.IGNORECASE)
        labels = [bool(seed_pattern.search(review.text)) for review in mining]
        positives = [review for review, label in zip(mining, labels) if label]
        hard_negatives = [
            review for review, label in zip(mining, labels)
            if not label and topic_pattern.search(review.text)
        ]
        anchors = mine_log_odds_anchors(texts, labels, min_df=args.min_ngram_document_frequency)
        category_seed = args.seed + category_index * 1009
        positive_examples = select_examples(positives, seed_pattern, embedder, category_seed)
        negative_examples = select_examples(hard_negatives, topic_pattern, embedder, category_seed + 1)
        positive_count = sum(labels)
        output[category["category_id"]] = {
            "template_type": category["template_type"],
            "attribute": category["attribute"],
            "seed_regex": category["seed_regex"],
            "lexical_anchors": anchors,
            "few_shot_positive": positive_examples,
            "few_shot_hard_negative": negative_examples,
            "threshold": None,
            "mining_stats": {
                "mining_set_size": len(mining),
                "proxy_positive_count": positive_count,
                "proxy_positive_rate": positive_count / len(mining),
                "hard_negative_candidate_count": len(hard_negatives),
                "low_sample_warning": positive_count < LOW_SAMPLE_CUTOFF,
            },
            "provenance": {
                "source_dataset": str(args.dataset.resolve()),
                "corpus_unique_size": len(reviews),
                "corpus_sample_size": len(mining) + len(holdout),
                "random_seed": args.seed,
                "embedding_backend": args.embedding_backend,
                "embedding_model": args.embedding_model if args.embedding_backend == "sentence-transformers" else None,
                "mined_at": mined_at,
            },
        }
    validate_semantic_dict(output)
    dict_path = args.output_dir / "semantic_dict.json"
    dict_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.output_dir / "semantic_dict_report.md", output)
    print_summary(output)
    return output


if __name__ == "__main__":
    run(parse_args())
