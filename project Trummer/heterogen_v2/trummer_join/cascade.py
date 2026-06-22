from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Iterable


BINARY_RE = re.compile(r"^\s*[\"']?(1|0|yes|no|true|false)\b", re.IGNORECASE)
PAIR_LABEL_RE = re.compile(r"\b(?:candidate|pair)[_ #:-]*(\d+)\b", re.IGNORECASE)
PLAIN_NUMBER_RE = re.compile(r"^\s*(\d+)\s*$")


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    movie: dict[str, str]
    review: dict[str, str]


@dataclass(frozen=True)
class Decision:
    candidate_id: int
    movie_id: str
    tconst: str
    score: float | None
    route: str
    error: str = ""


@dataclass(frozen=True)
class CascadeConfig:
    api_base: str = "http://127.0.0.1:11434"
    cheap_model: str = "gemma2:2b"
    expensive_model: str = "qwen2.5:3b"
    accept_threshold: float = 3.0
    reject_threshold: float = -1.5
    expensive_batch_size: int = 8
    request_timeout: float = 600.0
    max_review_chars: int = 1800

    def validate(self) -> None:
        if self.reject_threshold >= self.accept_threshold:
            raise ValueError("reject_threshold must be lower than accept_threshold")
        if self.expensive_batch_size < 1:
            raise ValueError("expensive_batch_size must be positive")


@dataclass
class CascadeMetrics:
    input_movies: int = 0
    input_reviews: int = 0
    candidate_pairs: int = 0
    cheap_calls: int = 0
    cheap_failures: int = 0
    cheap_early_accepts: int = 0
    cheap_early_rejects: int = 0
    expensive_candidates: int = 0
    expensive_calls: int = 0
    expensive_failures: int = 0
    expensive_accepts: int = 0
    elapsed_seconds: float = 0.0


class OllamaBinaryScorer:
    """Return log p(1)-log p(0), or a bounded score from a hard 1/0 answer."""

    def __init__(self, config: CascadeConfig) -> None:
        self.config = config
        self.api_base = config.api_base.rstrip("/")
        self._completions_supported: bool | None = None

    def score(self, candidate: Candidate, predicate: str) -> float:
        prompt = _pair_prompt(candidate, predicate, self.config.max_review_chars)
        completion_error: Exception | None = None
        if self._completions_supported is not False:
            try:
                payload = _post_json(
                    f"{self.api_base}/v1/completions",
                    {
                        "model": _plain_model(self.config.cheap_model),
                        "prompt": prompt,
                        "max_tokens": 8,
                        "temperature": 0,
                        "logprobs": 10,
                    },
                    self.config.request_timeout,
                )
                self._completions_supported = True
                return extract_binary_log_odds(payload)
            except urllib.error.HTTPError as exc:
                self._completions_supported = False
                completion_error = exc
            except Exception as exc:
                self._completions_supported = False
                completion_error = exc

        try:
            payload = _post_json(
                f"{self.api_base}/api/chat",
                {
                    "model": _plain_model(self.config.cheap_model),
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "integer", "enum": [0, 1]}
                        },
                        "required": ["answer"],
                    },
                    "options": {"temperature": 0, "num_predict": 32},
                },
                self.config.request_timeout,
            )
            return extract_binary_log_odds(payload)
        except Exception as native_error:
            raise RuntimeError(
                f"cheap scoring failed: completions={completion_error}; native={native_error}"
            ) from native_error


class OllamaExpensiveClassifier:
    """Classify an explicit batch of candidate pairs with the expensive model."""

    def __init__(self, config: CascadeConfig) -> None:
        self.config = config
        self.api_base = config.api_base.rstrip("/")

    def classify(self, candidates: list[Candidate], predicate: str) -> set[int]:
        if not candidates:
            return set()
        prompt = _batch_prompt(candidates, predicate, self.config.max_review_chars)
        request_payload = {
            "model": _plain_model(self.config.expensive_model),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "matching_pair_ids": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "enum": [candidate.candidate_id for candidate in candidates],
                        },
                    }
                },
                "required": ["matching_pair_ids"],
            },
            "options": {
                "temperature": 0,
                "num_predict": max(32, len(candidates) * 6),
            },
        }
        try:
            payload = _post_json(
                f"{self.api_base}/api/chat",
                request_payload,
                self.config.request_timeout,
            )
        except urllib.error.HTTPError:
            # Compatibility path for older Ollama versions without JSON-schema
            # structured output.
            request_payload.pop("format")
            payload = _post_json(
                f"{self.api_base}/api/chat",
                request_payload,
                self.config.request_timeout,
            )
        answer = str(payload.get("message", {}).get("content", ""))
        valid = {candidate.candidate_id for candidate in candidates}
        accepted: set[int] = set()
        try:
            structured = json.loads(answer)
            accepted.update(
                int(value)
                for value in structured.get("matching_pair_ids", [])
                if int(value) in valid
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        accepted.update(
            int(value)
            for value in PAIR_LABEL_RE.findall(answer)
            if int(value) in valid
        )
        for token in answer.split(","):
            match = PLAIN_NUMBER_RE.match(token)
            if match and int(match.group(1)) in valid:
                accepted.add(int(match.group(1)))

        # Some instruction-following models return the movie IDs even when
        # labels were requested. Accept that form only when the movie ID maps
        # to exactly one candidate in this batch; otherwise it is ambiguous.
        by_movie_id: dict[str, list[int]] = {}
        for candidate in candidates:
            by_movie_id.setdefault(
                str(candidate.movie.get("movie_id", "")),
                [],
            ).append(candidate.candidate_id)
        for movie_id, candidate_ids in by_movie_id.items():
            if movie_id and len(candidate_ids) == 1 and movie_id in answer:
                accepted.add(candidate_ids[0])
        return accepted


class CascadeJoin:
    def __init__(
        self,
        config: CascadeConfig,
        cheap_score: Callable[[Candidate, str], float] | None = None,
        expensive_classify: Callable[[list[Candidate], str], set[int]] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.cheap_score = cheap_score or OllamaBinaryScorer(config).score
        self.expensive_classify = (
            expensive_classify or OllamaExpensiveClassifier(config).classify
        )

    def run(
        self,
        movies: list[dict[str, str]],
        reviews: list[dict[str, str]],
        predicate: str,
    ) -> tuple[list[dict[str, str]], list[Decision], CascadeMetrics]:
        started = time.perf_counter()
        candidates = exact_id_candidates(movies, reviews)
        metrics = CascadeMetrics(
            input_movies=len(movies),
            input_reviews=len(reviews),
            candidate_pairs=len(candidates),
        )
        decisions: list[Decision] = []
        accepted: list[Candidate] = []
        uncertain: list[Candidate] = []

        for candidate in candidates:
            metrics.cheap_calls += 1
            try:
                score = float(self.cheap_score(candidate, predicate))
            except Exception as exc:
                metrics.cheap_failures += 1
                metrics.expensive_candidates += 1
                uncertain.append(candidate)
                decisions.append(_decision(candidate, None, "expensive", str(exc)))
                continue

            if score >= self.config.accept_threshold:
                metrics.cheap_early_accepts += 1
                accepted.append(candidate)
                decisions.append(_decision(candidate, score, "cheap_accept"))
            elif score <= self.config.reject_threshold:
                metrics.cheap_early_rejects += 1
                decisions.append(_decision(candidate, score, "cheap_reject"))
            else:
                metrics.expensive_candidates += 1
                uncertain.append(candidate)
                decisions.append(_decision(candidate, score, "expensive"))

        expensive_accepted: set[int] = set()
        for batch in _blocks(uncertain, self.config.expensive_batch_size):
            metrics.expensive_calls += 1
            try:
                expensive_accepted.update(self.expensive_classify(batch, predicate))
            except Exception:
                metrics.expensive_failures += 1
                continue

        by_id = {candidate.candidate_id: candidate for candidate in uncertain}
        accepted.extend(
            by_id[candidate_id]
            for candidate_id in sorted(expensive_accepted)
            if candidate_id in by_id
        )
        metrics.expensive_accepts = len(expensive_accepted & by_id.keys())
        metrics.elapsed_seconds = time.perf_counter() - started

        rows = [
            joined_row(
                candidate,
                "cheap_accept"
                if any(
                    decision.candidate_id == candidate.candidate_id
                    and decision.route == "cheap_accept"
                    for decision in decisions
                )
                else "expensive_accept",
            )
            for candidate in accepted
        ]
        return _deduplicate(rows), decisions, metrics


def exact_id_candidates(
    movies: Iterable[dict[str, str]],
    reviews: Iterable[dict[str, str]],
) -> list[Candidate]:
    movies_by_id: dict[str, list[dict[str, str]]] = {}
    for movie in movies:
        movies_by_id.setdefault(str(movie.get("movie_id", "")), []).append(movie)
    candidates: list[Candidate] = []
    for review in reviews:
        for movie in movies_by_id.get(str(review.get("tconst", "")), []):
            candidates.append(Candidate(len(candidates) + 1, movie, review))
    return candidates


def extract_binary_log_odds(payload: dict) -> float:
    values: dict[str, float] = {}
    for choice in payload.get("choices", []):
        logprobs = choice.get("logprobs") or {}
        for mapping in logprobs.get("top_logprobs") or []:
            if isinstance(mapping, dict):
                for token, logprob in mapping.items():
                    label = _binary_label(token)
                    if label is not None:
                        values[label] = float(logprob)
        text = choice.get("text")
        if text is not None and not values:
            label = _binary_label(text)
            if label is not None:
                return 2.0 if label == "1" else -2.0
    if "1" in values and "0" in values:
        return values["1"] - values["0"]
    message_content = payload.get("message", {}).get("content", "")
    if message_content:
        try:
            answer = json.loads(str(message_content)).get("answer")
            if answer in {0, 1, "0", "1"}:
                return 2.0 if str(answer) == "1" else -2.0
        except (AttributeError, json.JSONDecodeError):
            pass
        label = _binary_label(message_content)
        if label is not None:
            return 2.0 if label == "1" else -2.0
    label = _binary_label(payload.get("response", ""))
    if label is not None:
        return 2.0 if label == "1" else -2.0
    raise ValueError("model response did not contain a binary 1/0 decision")


def metrics_dict(metrics: CascadeMetrics) -> dict[str, int | float]:
    return asdict(metrics)


def joined_row(candidate: Candidate, source: str) -> dict[str, str]:
    movie, review = candidate.movie, candidate.review
    return {
        "candidate_id": str(candidate.candidate_id),
        "movie_id": movie.get("movie_id", ""),
        "title": movie.get("title", ""),
        "year": movie.get("year", ""),
        "director": movie.get("director", ""),
        "runtime": movie.get("runtime", ""),
        "genres": movie.get("genres", ""),
        "tconst": review.get("tconst", ""),
        "review": review.get("review", ""),
        "match_source": source,
    }


def _decision(
    candidate: Candidate,
    score: float | None,
    route: str,
    error: str = "",
) -> Decision:
    return Decision(
        candidate.candidate_id,
        candidate.movie.get("movie_id", ""),
        candidate.review.get("tconst", ""),
        score,
        route,
        error,
    )


def _pair_prompt(candidate: Candidate, predicate: str, max_review_chars: int) -> str:
    return (
        "Classify whether this candidate pair satisfies the predicate.\n"
        "Return exactly one token: 1 for yes, 0 for no.\n\n"
        f"Predicate: {predicate}\n"
        f"Movie: {candidate.movie.get('text', '')}\n"
        f"Review: {candidate.review.get('text', '')[:max_review_chars]}\n\n"
        "Answer:"
    )


def _batch_prompt(
    candidates: list[Candidate],
    predicate: str,
    max_review_chars: int,
) -> str:
    parts = [
        "Evaluate the explicit candidate pairs below.",
        f"Predicate: {predicate}",
        "Each pair has a label such as PAIR_12.",
        "Return only matching PAIR labels separated by commas, for example: PAIR_2, PAIR_7.",
        'Return "none" if no candidate satisfies it. Do not explain.',
        "",
    ]
    for candidate in candidates:
        parts.extend(
            [
                f"PAIR_{candidate.candidate_id}:",
                f"Movie: {candidate.movie.get('text', '')}",
                f"Review: {candidate.review.get('text', '')[:max_review_chars]}",
                "",
            ]
        )
    parts.append("Matching candidate IDs:")
    return "\n".join(parts)


def _binary_label(value: object) -> str | None:
    match = BINARY_RE.match(str(value))
    if not match:
        return None
    return "1" if match.group(1).lower() in {"1", "yes", "true"} else "0"


def _plain_model(model: str) -> str:
    return model.removeprefix("ollama/")


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _blocks(rows: list[Candidate], size: int) -> Iterable[list[Candidate]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result = []
    for row in rows:
        key = (row["movie_id"], row["tconst"], row["review"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result
