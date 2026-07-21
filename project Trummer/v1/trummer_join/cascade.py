from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Iterable


PAIR_LABEL_RE = re.compile(r"\b(?:candidate|pair)[_ #:-]*(\d+)\b", re.IGNORECASE)
PLAIN_NUMBER_RE = re.compile(r"^\s*(\d+)\s*$")

RECALL_INSTRUCTIONS = """Determine whether the review contains credible evidence for the condition.

Maximize recall:
- Answer YES if direct, indirect, synonymous, or reasonably implied evidence exists.
- Answer NO only when the condition is clearly absent or contradicted.
- If the evidence is genuinely ambiguous, answer UNCERTAIN.
- Do not reject merely because the review uses different wording."""

EXPENSIVE_THINK = os.environ.get("EXPENSIVE_THINK", "0") == "1"
EXPENSIVE_NUM_PREDICT = int(
    os.environ.get("EXPENSIVE_NUM_PREDICT", "512" if EXPENSIVE_THINK else "128")
)


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
    cheap_model: str = "gemma4:e2b"
    expensive_model: str = "gemma4:e4b"
    cascade_target: float = 0.9
    calibration_budget: int = 20
    manual_confidence_threshold: float | None = None
    cheap_batch_size: int = 8
    expensive_batch_size: int = 32
    request_timeout: float = 600.0
    max_review_chars: int = 1800

    def validate(self) -> None:
        if not 0.0 < self.cascade_target <= 1.0:
            raise ValueError("cascade_target must be in (0, 1]")
        if self.calibration_budget < 0:
            raise ValueError("calibration_budget must be non-negative")
        if (
            self.manual_confidence_threshold is not None
            and self.manual_confidence_threshold < 0
        ):
            raise ValueError("manual_confidence_threshold must be non-negative")
        if self.cheap_batch_size < 1:
            raise ValueError("cheap_batch_size must be positive")
        if self.expensive_batch_size < 1:
            raise ValueError("expensive_batch_size must be positive")
        if self.expensive_batch_size <= self.cheap_batch_size:
            raise ValueError(
                "expensive_batch_size must exceed cheap_batch_size so fallback "
                "calls can be coalesced across cheap batches"
            )


@dataclass
class CascadeMetrics:
    input_movies: int = 0
    input_reviews: int = 0
    candidate_pairs: int = 0
    cheap_batches: int = 0
    cheap_calls: int = 0
    cheap_failures: int = 0
    cheap_failure_candidates: int = 0
    cheap_early_accepts: int = 0
    cheap_early_rejects: int = 0
    calibration_candidates: int = 0
    calibration_expensive_calls: int = 0
    calibration_expensive_accepts: int = 0
    calibration_agreement: float = 0.0
    learned_confidence_threshold: float | None = None
    routing_confidence_threshold: float | None = None
    manual_confidence_threshold: float | None = None
    expensive_candidates: int = 0
    expensive_calls: int = 0
    fallback_expensive_calls: int = 0
    expensive_failures: int = 0
    expensive_accepts: int = 0
    nonempty_fallback_rows: int = 0
    cheap_seconds: float = 0.0
    expensive_seconds: float = 0.0
    cheap_time_percent: float = 0.0
    expensive_time_percent: float = 0.0
    elapsed_seconds: float = 0.0


class OllamaBatchBinaryScorer:
    """Classify one candidate batch in one cheap-model request.

    Ollama's structured chat API returns hard binary decisions, represented as
    -2/+2 so the cascade can derive proxy labels and confidence values.
    """

    def __init__(self, config: CascadeConfig) -> None:
        self.config = config
        self.api_base = config.api_base.rstrip("/")

    def score_batch(
        self,
        candidates: list[Candidate],
        predicate: str,
    ) -> dict[int, float]:
        if not candidates:
            return {}
        prompt = _cheap_batch_prompt(
            candidates,
            predicate,
            self.config.max_review_chars,
        )
        ids = [candidate.candidate_id for candidate in candidates]
        request_payload = {
            "model": _plain_model(self.config.cheap_model),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": {
                "type": "object",
                "properties": {
                    "decisions": {
                        "type": "array",
                        "minItems": len(ids),
                        "maxItems": len(ids),
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {
                                    "type": "integer",
                                    "enum": ids,
                                },
                                "decision": {"type": "string", "enum": ["yes", "no", "uncertain"]},
                            },
                            "required": ["candidate_id", "decision"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["decisions"],
                "additionalProperties": False,
            },
            "options": {
                "temperature": 0,
                "num_predict": max(128, 32 + 24 * len(ids)),
                "num_ctx": 4096,
            },
        }
        trace = os.environ.get("PIPELINE_TRACE", "0") == "1"
        if trace:
            print(f"[TRACE V3_2 CHEAP PROMPT]\n{prompt}\n[TRACE END]", flush=True)
        payload = _post_json(
            f"{self.api_base}/api/chat",
            request_payload,
            self.config.request_timeout,
        )
        if trace:
            print(f"[TRACE V3_2 CHEAP OUTPUT] {payload!r}", flush=True)
        return _parse_batch_scores(payload, set(ids))


class OllamaExpensiveClassifier:
    """Classify uncertain candidates in coalesced expensive-model batches."""

    def __init__(self, config: CascadeConfig) -> None:
        self.config = config
        self.api_base = config.api_base.rstrip("/")

    def classify(self, candidates: list[Candidate], predicate: str) -> set[int]:
        if not candidates:
            return set()
        prompt = _expensive_batch_prompt(
            candidates,
            predicate,
            self.config.max_review_chars,
        )
        valid = {candidate.candidate_id for candidate in candidates}
        request_payload = {
            "model": _plain_model(self.config.expensive_model),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": EXPENSIVE_THINK,
            "format": {
                "type": "object",
                "properties": {
                    "decisions": {
                        "type": "array",
                        "minItems": len(valid),
                        "maxItems": len(valid),
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "integer", "enum": sorted(valid)},
                                "decision": {"type": "string", "enum": ["yes", "no", "uncertain"]},
                            },
                            "required": ["candidate_id", "decision"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["decisions"],
                "additionalProperties": False,
            },
            "options": {
                "temperature": 0,
                "num_predict": max(EXPENSIVE_NUM_PREDICT, 32 + 24 * len(valid)),
                "num_ctx": 4096,
            },
        }
        trace = os.environ.get("PIPELINE_TRACE", "0") == "1"
        if trace:
            print(f"[TRACE V3_2 EXPENSIVE PROMPT]\n{prompt}\n[TRACE END]", flush=True)
        payload = _post_json(
            f"{self.api_base}/api/chat",
            request_payload,
            self.config.request_timeout,
        )
        answer = str(payload.get("message", {}).get("content", ""))
        if trace:
            print(f"[TRACE V3_2 EXPENSIVE OUTPUT] {payload!r}", flush=True)
        if not _has_complete_decision_coverage(answer, valid):
            raise ValueError(
                "expensive response did not contain exactly one valid decision "
                "for every candidate: " + answer[:200]
            )
        return _parse_matching_ids(answer, candidates, valid)


class CascadeJoin:
    def __init__(
        self,
        config: CascadeConfig,
        cheap_score_batch: Callable[
            [list[Candidate], str], dict[int, float]
        ] | None = None,
        expensive_classify: Callable[
            [list[Candidate], str], set[int]
        ] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.cheap_score_batch = (
            cheap_score_batch or OllamaBatchBinaryScorer(config).score_batch
        )
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
        scored: list[tuple[Candidate, float | None, str]] = []

        cheap_batches = list(_blocks(candidates, self.config.cheap_batch_size))
        cheap_started = time.perf_counter()
        for batch_index, batch in enumerate(cheap_batches, 1):
            metrics.cheap_batches += 1
            metrics.cheap_calls += 1
            call_started = time.perf_counter()
            try:
                scores = self.cheap_score_batch(batch, predicate)
            except Exception as exc:
                metrics.cheap_seconds += time.perf_counter() - call_started
                metrics.cheap_failures += 1
                metrics.cheap_failure_candidates += len(batch)
                scored.extend((candidate, None, str(exc)) for candidate in batch)
                _print_progress(
                    "Batch-wise cascade cheap scoring",
                    batch_index,
                    len(cheap_batches),
                    cheap_started,
                )
                continue
            metrics.cheap_seconds += time.perf_counter() - call_started
            _print_progress(
                "Batch-wise cascade cheap scoring",
                batch_index,
                len(cheap_batches),
                cheap_started,
            )

            for candidate in batch:
                score = scores.get(candidate.candidate_id)
                if score is None:
                    metrics.cheap_failure_candidates += 1
                    scored.append(
                        (candidate, None, "cheap batch omitted candidate")
                    )
                else:
                    scored.append((candidate, float(score), ""))

        if self.config.manual_confidence_threshold is None:
            learned_threshold = self._learn_confidence_threshold(scored, predicate, metrics)
            routing_threshold = learned_threshold
        else:
            learned_threshold = None
            routing_threshold = self.config.manual_confidence_threshold
        metrics.learned_confidence_threshold = learned_threshold
        metrics.routing_confidence_threshold = routing_threshold
        metrics.manual_confidence_threshold = self.config.manual_confidence_threshold

        for candidate, score, error in scored:
            if score is None:
                metrics.expensive_candidates += 1
                uncertain.append(candidate)
                decisions.append(_decision(candidate, None, "expensive", error))
            elif _is_confident(score, routing_threshold) and score >= 0:
                metrics.cheap_early_accepts += 1
                accepted.append(candidate)
                decisions.append(_decision(candidate, score, "cheap_accept"))
            elif _is_confident(score, routing_threshold):
                metrics.cheap_early_rejects += 1
                decisions.append(_decision(candidate, score, "cheap_reject"))
            else:
                metrics.expensive_candidates += 1
                uncertain.append(candidate)
                decisions.append(_decision(candidate, score, "expensive"))

        expensive_accepted: set[int] = set()
        fallback_batches = list(_blocks(uncertain, self.config.expensive_batch_size))
        fallback_started = time.perf_counter()
        for batch_index, batch in enumerate(fallback_batches, 1):
            metrics.expensive_calls += 1
            metrics.fallback_expensive_calls += 1
            call_started = time.perf_counter()
            try:
                expensive_accepted.update(
                    self.expensive_classify(batch, predicate)
                )
            except Exception:
                metrics.expensive_seconds += time.perf_counter() - call_started
                metrics.expensive_failures += 1
                _print_progress(
                    "Batch-wise cascade expensive fallback",
                    batch_index,
                    len(fallback_batches),
                    fallback_started,
                )
                continue
            metrics.expensive_seconds += time.perf_counter() - call_started
            _print_progress(
                "Batch-wise cascade expensive fallback",
                batch_index,
                len(fallback_batches),
                fallback_started,
            )

        by_id = {candidate.candidate_id: candidate for candidate in uncertain}
        accepted.extend(
            by_id[candidate_id]
            for candidate_id in sorted(expensive_accepted)
            if candidate_id in by_id
        )
        metrics.expensive_accepts = len(expensive_accepted & by_id.keys())
        metrics.elapsed_seconds = time.perf_counter() - started
        _set_time_percentages(metrics)

        cheap_accepted_ids = {
            decision.candidate_id
            for decision in decisions
            if decision.route == "cheap_accept"
        }
        rows = [
            joined_row(
                candidate,
                "cheap_accept"
                if candidate.candidate_id in cheap_accepted_ids
                else "expensive_accept",
            )
            for candidate in accepted
        ]
        if not rows and candidates:
            scored_candidates = [
                (score, candidate)
                for candidate, score, _error in scored
                if score is not None
            ]
            fallback = (
                max(scored_candidates, key=lambda item: item[0])[1]
                if scored_candidates
                else candidates[0]
            )
            rows = [joined_row(fallback, "nonempty_fallback")]
            metrics.nonempty_fallback_rows = 1
        return _deduplicate(rows), decisions, metrics

    def _learn_confidence_threshold(
        self,
        scored: list[tuple[Candidate, float | None, str]],
        predicate: str,
        metrics: CascadeMetrics,
    ) -> float | None:
        calibration_candidates = _calibration_sample(
            scored,
            self.config.calibration_budget,
        )
        if not calibration_candidates:
            return None
        scores_by_id = {
            candidate.candidate_id: score
            for candidate, score, _ in scored
            if score is not None
        }
        records: list[tuple[float, bool, bool]] = []
        calibration_batches = list(
            _blocks(calibration_candidates, self.config.expensive_batch_size)
        )
        calibration_started = time.perf_counter()
        for batch_index, batch in enumerate(calibration_batches, 1):
            metrics.calibration_candidates += len(batch)
            metrics.calibration_expensive_calls += 1
            metrics.expensive_calls += 1
            call_started = time.perf_counter()
            try:
                oracle_accepts = self.expensive_classify(batch, predicate)
            except Exception:
                metrics.expensive_seconds += time.perf_counter() - call_started
                metrics.expensive_failures += 1
                _print_progress(
                    "Batch-wise cascade threshold calibration",
                    batch_index,
                    len(calibration_batches),
                    calibration_started,
                )
                continue
            metrics.expensive_seconds += time.perf_counter() - call_started
            _print_progress(
                "Batch-wise cascade threshold calibration",
                batch_index,
                len(calibration_batches),
                calibration_started,
            )
            metrics.calibration_expensive_accepts += len(oracle_accepts)
            for candidate in batch:
                score = scores_by_id.get(candidate.candidate_id)
                if score is None:
                    continue
                records.append(
                    (
                        abs(score),
                        score >= 0,
                        candidate.candidate_id in oracle_accepts,
                    )
                )
        threshold, agreement = _learn_threshold(records, self.config.cascade_target)
        metrics.calibration_agreement = agreement
        return threshold


def exact_id_candidates(
    movies: Iterable[dict[str, str]],
    reviews: Iterable[dict[str, str]],
) -> list[Candidate]:
    movies_by_id: dict[str, list[dict[str, str]]] = {}
    for movie in movies:
        movies_by_id.setdefault(
            str(movie.get("movie_id", "")),
            [],
        ).append(movie)
    candidates: list[Candidate] = []
    for review in reviews:
        for movie in movies_by_id.get(str(review.get("tconst", "")), []):
            candidates.append(
                Candidate(len(candidates) + 1, movie, review)
            )
    return candidates


def metrics_dict(metrics: CascadeMetrics) -> dict[str, object]:
    return asdict(metrics)


def _print_progress(label: str, done: int, total: int, started: float) -> None:
    if total <= 0:
        return
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    percent = 100.0 * done / total
    width = 24
    filled = min(width, int(width * done / total))
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"{label}: [{bar}] {done}/{total} ({percent:.1f}%) "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
        flush=True,
    )


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


def _set_time_percentages(metrics: CascadeMetrics) -> None:
    model_seconds = metrics.cheap_seconds + metrics.expensive_seconds
    if model_seconds <= 0:
        return
    metrics.cheap_time_percent = 100.0 * metrics.cheap_seconds / model_seconds
    metrics.expensive_time_percent = (
        100.0 * metrics.expensive_seconds / model_seconds
    )


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


def _cheap_batch_prompt(
    candidates: list[Candidate],
    predicate: str,
    max_review_chars: int,
) -> str:
    parts = [
        "Classify every explicit candidate pair below.",
        f"Predicate: {predicate}",
        RECALL_INSTRUCTIONS,
        "For each candidate return only candidate_id and decision.",
        "Do not include confidence, evidence, explanations, or prose.",
        "Return every candidate exactly once and do not explain.",
        "",
    ]
    for candidate in candidates:
        parts.extend(
            [
                f"CANDIDATE_{candidate.candidate_id}:",
                f"Movie: {candidate.movie.get('text', '')}",
                f"Review: {candidate.review.get('text', '')[:max_review_chars]}",
                "",
            ]
        )
    parts.append("JSON decisions:")
    return "\n".join(parts)


def _expensive_batch_prompt(
    candidates: list[Candidate],
    predicate: str,
    max_review_chars: int,
) -> str:
    parts = [
        "Evaluate the explicit candidate pairs below.",
        f"Predicate: {predicate}",
        RECALL_INSTRUCTIONS,
        "For each candidate return only candidate_id and decision.",
        "Do not include confidence, evidence, explanations, or prose.",
        "Return every candidate exactly once and do not add prose outside the JSON.",
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
    parts.append("JSON decisions:")
    return "\n".join(parts)


def _parse_batch_scores(
    payload: dict,
    valid: set[int],
) -> dict[int, float]:
    answer = str(payload.get("message", {}).get("content", ""))
    parsed = _extract_json(answer)
    if parsed is None:
        raise ValueError(
            "cheap batch response was not valid JSON: " + answer[:200]
        )
    scores: dict[int, float] = {}
    decisions: object
    if isinstance(parsed, dict):
        decisions = parsed.get(
            "decisions",
            parsed.get("answers", parsed.get("results", parsed)),
        )
    else:
        decisions = parsed
    if isinstance(decisions, dict):
        decisions = [
            {"candidate_id": candidate_id, "answer": answer_value}
            for candidate_id, answer_value in decisions.items()
        ]
    if not isinstance(decisions, list):
        decisions = []
    for item in decisions:
        if not isinstance(item, dict):
            continue
        try:
            candidate_id = int(
                item.get("candidate_id", item.get("pair_id", item.get("id")))
            )
        except (KeyError, TypeError, ValueError):
            continue
        label = _binary_answer(
            item.get(
                "answer",
                item.get("label", item.get("decision", item.get("match"))),
            )
        )
        if candidate_id in valid and label is not None:
            scores[candidate_id] = 2.0 if label == "1" else -2.0
    if set(scores) != valid:
        missing = sorted(valid - set(scores))
        raise ValueError(f"cheap batch response omitted candidate IDs: {missing}")
    return scores


def _parse_matching_ids(
    answer: str,
    candidates: list[Candidate],
    valid: set[int],
) -> set[int]:
    accepted: set[int] = set()
    structured = _extract_json(answer)
    if isinstance(structured, dict):
        decisions = structured.get("decisions", [])
        if isinstance(decisions, list):
            for item in decisions:
                if not isinstance(item, dict):
                    continue
                try:
                    candidate_id = int(item.get("candidate_id"))
                except (TypeError, ValueError):
                    continue
                if (
                    candidate_id in valid
                    and str(item.get("decision", "")).strip().lower() == "yes"
                ):
                    accepted.add(candidate_id)
        for key in (
            "matching_pair_ids",
            "matching_candidate_ids",
            "pair_ids",
            "matches",
        ):
            values = structured.get(key, [])
            if isinstance(values, list):
                for value in values:
                    try:
                        candidate_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    if candidate_id in valid:
                        accepted.add(candidate_id)
    accepted.update(
        int(value)
        for value in PAIR_LABEL_RE.findall(answer)
        if int(value) in valid
    )
    for token in answer.split(","):
        match = PLAIN_NUMBER_RE.match(token)
        if match and int(match.group(1)) in valid:
            accepted.add(int(match.group(1)))
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


def _binary_answer(value: object) -> str | None:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and value in (0, 1):
        return str(int(value))
    text = str(value).strip().lower()
    if text in {"1", "yes", "true", "match", "matched"}:
        return "1"
    if text in {"0", "no", "false", "no match", "not matched"}:
        return "0"
    return None


def _extract_json(text: str) -> object | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _is_valid_decision_answer(answer: str) -> bool:
    parsed = _extract_json(answer)
    if isinstance(parsed, dict):
        decisions = parsed.get("decisions")
        if isinstance(decisions, list) and decisions:
            for item in decisions:
                if not isinstance(item, dict):
                    return False
                try:
                    int(item["candidate_id"])
                except (KeyError, TypeError, ValueError):
                    return False
                if (
                    str(item.get("decision", "")).strip().lower()
                    not in {"yes", "no", "uncertain"}
                ):
                    return False
            return True
        return any(
            key in parsed
            for key in (
                "matching_pair_ids",
                "matching_candidate_ids",
                "pair_ids",
                "matches",
            )
        )
    if parsed == []:
        return True
    return bool(
        re.search(
            r"\b(?:none|no matches|no matching (?:pairs|candidates))\b",
            answer,
            re.IGNORECASE,
        )
    )


def _has_complete_decision_coverage(answer: str, valid: set[int]) -> bool:
    parsed = _extract_json(answer)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
        return False
    returned: list[int] = []
    for item in parsed["decisions"]:
        if not isinstance(item, dict):
            return False
        try:
            candidate_id = int(item["candidate_id"])
        except (KeyError, TypeError, ValueError):
            return False
        if str(item.get("decision", "")).strip().lower() not in {"yes", "no", "uncertain"}:
            return False
        returned.append(candidate_id)
    return len(returned) == len(valid) and set(returned) == valid


def _plain_model(model: str) -> str:
    return model.removeprefix("ollama/")


def _calibration_sample(
    scored: list[tuple[Candidate, float | None, str]],
    budget: int,
) -> list[Candidate]:
    if budget <= 0:
        return []
    eligible = [
        (candidate, abs(score))
        for candidate, score, _ in scored
        if score is not None
    ]
    if len(eligible) <= budget:
        return [candidate for candidate, _ in eligible]
    ranked = sorted(eligible, key=lambda item: (-item[1], item[0].candidate_id))
    if budget == 1:
        return [ranked[0][0]]
    selected: list[Candidate] = []
    selected_ids: set[int] = set()
    for index in range(budget):
        rank = round(index * (len(ranked) - 1) / (budget - 1))
        candidate = ranked[rank][0]
        if candidate.candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.candidate_id)
    return selected


def _learn_threshold(
    records: list[tuple[float, bool, bool]],
    target: float,
) -> tuple[float | None, float]:
    best_agreement = 0.0
    for threshold in sorted({confidence for confidence, _, _ in records}):
        selected = [
            (proxy_label, oracle_label)
            for confidence, proxy_label, oracle_label in records
            if confidence >= threshold
        ]
        if not selected:
            continue
        agreement = sum(
            int(proxy_label == oracle_label)
            for proxy_label, oracle_label in selected
        ) / len(selected)
        best_agreement = max(best_agreement, agreement)
        if agreement >= target:
            return threshold, agreement
    return None, best_agreement


def _is_confident(score: float, threshold: float | None) -> bool:
    return threshold is not None and abs(score) >= threshold


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
