from __future__ import annotations

import math
import os
from typing import Iterable

import httpx


DEFAULT_MODEL = os.environ.get("SUQL_MODEL", "ollama/phi4-mini")
DEFAULT_API_BASE = os.environ.get("SUQL_API_BASE", "http://localhost:11434")


def _label(token: str) -> str | None:
    value = token.strip().strip('"').strip("'").lower().rstrip(".")
    first = value.split()[0].rstrip(".,:;") if value.split() else value
    if value in {"1", "yes", "true"} or first in {"1", "yes", "true"}:
        return "1"
    if value in {"0", "no", "false"} or first in {"0", "no", "false"}:
        return "0"
    return None


def _single_score(label: str, logprob: float | None) -> float:
    if logprob is None:
        magnitude = 2.0
    else:
        probability = min(max(math.exp(float(logprob)), 0.500001), 0.999999)
        magnitude = math.log(probability) - math.log1p(-probability)
    return magnitude if label == "1" else -magnitude


def _mapping_score(mapping: dict) -> float | None:
    values: dict[str, float] = {}
    for token, logprob in mapping.items():
        label = _label(str(token))
        if label is not None:
            values[label] = float(logprob)
    if "1" in values and "0" in values:
        return values["1"] - values["0"]
    for token, logprob in mapping.items():
        label = _label(str(token))
        if label is not None:
            return _single_score(label, float(logprob))
    return None


def _list_score(items: Iterable[dict]) -> float | None:
    values: dict[str, float] = {}
    materialized = list(items)
    for item in materialized:
        label = _label(str(item.get("token", "")))
        if label is not None and "logprob" in item:
            values[label] = float(item["logprob"])
    if "1" in values and "0" in values:
        return values["1"] - values["0"]
    for item in materialized:
        label = _label(str(item.get("token", "")))
        if label is not None:
            value = item.get("logprob")
            return _single_score(label, float(value) if value is not None else None)
    return None


def extract_binary_log_odds(payload: dict) -> float:
    candidates: list[object] = []
    for choice in payload.get("choices", []):
        logprobs = choice.get("logprobs") or {}
        candidates.append(logprobs)
        if isinstance(logprobs, dict):
            candidates.extend(logprobs.get("top_logprobs") or [])
            for item in logprobs.get("content") or []:
                if isinstance(item, dict):
                    candidates.append(item.get("top_logprobs"))
    candidates.extend([payload.get("logprobs"), payload.get("response_logprobs")])
    if isinstance(payload.get("logprobs"), list):
        candidates.extend(payload["logprobs"])
    for candidate in candidates:
        if isinstance(candidate, dict):
            score = _mapping_score(candidate)
            if score is not None:
                return score
            for item in candidate.get("top_logprobs") or []:
                if isinstance(item, dict):
                    score = _mapping_score(item)
                    if score is not None:
                        return score
        elif isinstance(candidate, list):
            score = _list_score(item for item in candidate if isinstance(item, dict))
            if score is not None:
                return score
    response_label = _label(str(payload.get("response", "")))
    if response_label is not None:
        return _single_score(response_label, None)
    for choice in payload.get("choices", []):
        text = choice.get("text")
        if text is None and isinstance(choice.get("message"), dict):
            text = choice["message"].get("content")
        choice_label = _label(str(text or ""))
        if choice_label is not None:
            return _single_score(choice_label, None)
    raise ValueError("Could not extract a binary score from the model response.")


class OllamaLogOddsScorer:
    def __init__(self, model: str | None = None, api_base: str | None = None, timeout: float = 120.0) -> None:
        self.model = model or DEFAULT_MODEL
        self.ollama_model = self.model.removeprefix("ollama/")
        self.api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
        self.timeout = timeout
        self._openai_completions_available: bool | None = None

    def score(self, review: str, question: str) -> float:
        prompt = self._prompt(review, question)
        openai_error: Exception | None = None
        if self._openai_completions_available is not False:
            try:
                response = httpx.post(
                    f"{self.api_base}/v1/completions",
                    json={"model": self.ollama_model, "prompt": prompt, "max_tokens": 1,
                          "temperature": 0, "logprobs": 10, "think": False},
                    timeout=self.timeout,
                )
                if response.status_code == 404:
                    self._openai_completions_available = False
                else:
                    response.raise_for_status()
                    score = extract_binary_log_odds(response.json())
                    self._openai_completions_available = True
                    return score
            except Exception as exc:
                openai_error = exc
                self._openai_completions_available = False
        try:
            response = httpx.post(
                f"{self.api_base}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False,
                      "think": False, "options": {"temperature": 0, "num_predict": 1}},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return extract_binary_log_odds(response.json())
        except Exception as native_error:
            if openai_error is not None:
                raise RuntimeError(
                    f"Both scoring APIs failed: {openai_error}; {native_error}"
                ) from native_error
            raise

    @staticmethod
    def _prompt(review: str, question: str) -> str:
        return (
            "Classify whether the review provides evidence for a Yes answer.\n"
            "This is recall-first classification: avoid false negatives. If there is credible "
            "supporting evidence or ambiguity, answer 1. Answer 0 only when clearly absent.\n"
            "Return exactly one token: 1 for Yes, 0 for No.\n\n"
            f"Question: {question}\nReview: {review[:1800]}\n\nAnswer:"
        )


def parse_label(value: object) -> int:
    label = _label(str(value))
    if label is None:
        raise ValueError(f"Unsupported label value: {value!r}")
    return int(label)
