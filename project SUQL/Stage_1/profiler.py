from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Iterable

import httpx
import numpy as np
import pandas as pd


DEFAULT_MODEL = os.environ.get("SUQL_MODEL", "ollama/phi4-mini")
DEFAULT_API_BASE = os.environ.get("SUQL_API_BASE", "http://localhost:11434")


def ollama_model_name(model: str) -> str:
    """Return the model name expected by Ollama's native/OpenAI-compatible API."""
    return model.removeprefix("ollama/")


@dataclasses.dataclass(frozen=True)
class LabelledExample:
    review: str
    question: str
    label: int


@dataclasses.dataclass(frozen=True)
class ThresholdResult:
    question: str
    accept_threshold: float
    reject_threshold: float
    recall_lower: float
    precision_lower: float
    recall_target: float
    precision_target: float
    credible_level: float
    tp: int
    fp: int
    fn: int
    tn: int
    labelled_examples: int
    corpus_examples: int
    accept_count: int
    selectivity: float

    def to_json_dict(self) -> dict:
        return dataclasses.asdict(self)


class OllamaLogOddsScorer:
    """Score answer(review, question) with log p('1') - log p('0').

    Ollama's native /api/generate endpoint may return only the generated token's
    logprob instead of top-logprobs for both binary labels. In that case we
    derive a signed confidence score from the generated 1/0 token, which keeps
    the runtime filter usable while preserving real log-odds when available.
    """

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.ollama_model = ollama_model_name(self.model)
        self.api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
        self.timeout = timeout
        self._openai_completions_available: bool | None = None

    def score(self, review: str, question: str) -> float:
        prompt = self._prompt(review, question)
        openai_error: Exception | None = None
        if self._openai_completions_available is not False:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": 10,
            }
            try:
                response = httpx.post(f"{self.api_base}/v1/completions", json=payload, timeout=self.timeout)
                if response.status_code == 404:
                    self._openai_completions_available = False
                else:
                    response.raise_for_status()
                    try:
                        score = extract_binary_log_odds(response.json())
                    except Exception as exc:
                        openai_error = exc
                        self._openai_completions_available = False
                    else:
                        self._openai_completions_available = True
                        return score
            except Exception as exc:
                openai_error = exc
                self._openai_completions_available = False

        native_payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1},
        }
        native_response = httpx.post(
            f"{self.api_base}/api/generate",
            json=native_payload,
            timeout=self.timeout,
        )
        try:
            native_response.raise_for_status()
            return extract_binary_log_odds(native_response.json())
        except Exception as native_error:
            if openai_error is not None:
                raise RuntimeError(
                    "OpenAI-compatible scoring failed; native Ollama scoring also failed. "
                    f"OpenAI-compatible error: {type(openai_error).__name__}: {openai_error}. "
                    f"Native error: {type(native_error).__name__}: {native_error}"
                ) from native_error
            raise

    @staticmethod
    def _prompt(review: str, question: str) -> str:
        return (
            "Classify whether the review answers the question with Yes.\n"
            "Return exactly one token: 1 for Yes, 0 for No.\n\n"
            f"Question: {question}\n"
            f"Review: {review[:1800]}\n\n"
            "Answer:"
        )


def _normalise_token(token: str) -> str:
    return token.strip().strip('"').strip("'")


def _label_from_token(token: str) -> str | None:
    normalised = _normalise_token(token).lower().rstrip(".")
    first_token = normalised.split()[0].rstrip(".,:;") if normalised.split() else normalised
    if normalised in {"1", "yes", "true"} or first_token in {"1", "yes", "true"}:
        return "1"
    if normalised in {"0", "no", "false"} or first_token in {"0", "no", "false"}:
        return "0"
    return None


def _score_from_single_label(label: str, logprob: float | None) -> float:
    if logprob is None:
        magnitude = 2.0
    else:
        eps = 1e-6
        probability = min(max(math.exp(float(logprob)), 0.5 + eps), 1.0 - eps)
        magnitude = math.log(probability) - math.log1p(-probability)
    return magnitude if label == "1" else -magnitude


def _mapping_log_odds(mapping: dict) -> float | None:
    values: dict[str, float] = {}
    for token, logprob in mapping.items():
        label = _label_from_token(str(token))
        if label is not None:
            values[label] = float(logprob)
    if "1" in values and "0" in values:
        return values["1"] - values["0"]
    return None


def _mapping_single_label_score(mapping: dict) -> float | None:
    for token, logprob in mapping.items():
        label = _label_from_token(str(token))
        if label is not None:
            return _score_from_single_label(label, float(logprob))
    return None


def _list_log_odds(items: Iterable[dict]) -> float | None:
    values: dict[str, float] = {}
    for item in items:
        label = _label_from_token(str(item.get("token", "")))
        if label is not None and "logprob" in item:
            values[label] = float(item["logprob"])
    if "1" in values and "0" in values:
        return values["1"] - values["0"]
    return None


def _list_single_label_score(items: Iterable[dict]) -> float | None:
    for item in items:
        label = _label_from_token(str(item.get("token", "")))
        if label is not None:
            logprob = item.get("logprob")
            return _score_from_single_label(label, float(logprob) if logprob is not None else None)
    return None


def _response_single_label_score(payload: dict) -> float | None:
    response = payload.get("response")
    if response is None:
        return None
    label = _label_from_token(str(response))
    if label is None:
        return None
    return _score_from_single_label(label, None)


def _choices_text_single_label_score(payload: dict) -> float | None:
    for choice in payload.get("choices", []):
        if not isinstance(choice, dict):
            continue
        text = choice.get("text")
        if text is None:
            message = choice.get("message")
            if isinstance(message, dict):
                text = message.get("content")
        if text is None:
            continue
        label = _label_from_token(str(text))
        if label is not None:
            return _score_from_single_label(label, None)
    return None


def extract_binary_log_odds(payload: dict) -> float:
    """
    Extract log p('1') - log p('0') from common OpenAI/Ollama logprob shapes.
    Raises if the model response does not include both token logprobs.
    """
    candidates: list[object] = []

    for choice in payload.get("choices", []):
        logprobs = choice.get("logprobs") or {}
        candidates.append(logprobs)
        if isinstance(logprobs, dict):
            candidates.extend(logprobs.get("top_logprobs") or [])
            content = logprobs.get("content") or []
            if content:
                candidates.extend(item.get("top_logprobs") for item in content if isinstance(item, dict))

    candidates.append(payload.get("logprobs"))
    candidates.append(payload.get("response_logprobs"))
    if isinstance(payload.get("logprobs"), list):
        candidates.extend(payload["logprobs"])

    for candidate in candidates:
        if isinstance(candidate, dict):
            diff = _mapping_log_odds(candidate)
            if diff is not None:
                return diff
            top = candidate.get("top_logprobs")
            if isinstance(top, list):
                for item in top:
                    if isinstance(item, dict):
                        diff = _mapping_log_odds(item)
                        if diff is not None:
                            return diff
        elif isinstance(candidate, list):
            if candidate and all(isinstance(item, dict) and "token" in item for item in candidate):
                diff = _list_log_odds(candidate)
                if diff is not None:
                    return diff
            for item in candidate:
                if isinstance(item, dict):
                    diff = _mapping_log_odds(item)
                    if diff is not None:
                        return diff

    for candidate in candidates:
        if isinstance(candidate, dict):
            score = _mapping_single_label_score(candidate)
            if score is not None:
                return score
            top = candidate.get("top_logprobs")
            if isinstance(top, list):
                for item in top:
                    if isinstance(item, dict):
                        score = _mapping_single_label_score(item)
                        if score is not None:
                            return score
        elif isinstance(candidate, list):
            if candidate and all(isinstance(item, dict) and "token" in item for item in candidate):
                score = _list_single_label_score(candidate)
                if score is not None:
                    return score
            for item in candidate:
                if isinstance(item, dict):
                    score = _mapping_single_label_score(item)
                    if score is not None:
                        return score

    response_score = _response_single_label_score(payload)
    if response_score is not None:
        return response_score

    choices_text_score = _choices_text_single_label_score(payload)
    if choices_text_score is not None:
        return choices_text_score

    raise ValueError("Could not find logprobs for both '1' and '0' tokens in Ollama response.")


def parse_label(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "yes", "true", "y", "positive"}:
        return 1
    if text in {"0", "no", "false", "n", "negative"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def beta_lower_bound(successes: int, failures: int, credible_level: float) -> float:
    """Lower credible bound for Beta(1+successes, 1+failures)."""
    from scipy.stats import beta

    tail_probability = 1.0 - credible_level
    return float(beta.ppf(tail_probability, 1 + successes, 1 + failures))


def confusion_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    predicted = scores >= threshold
    actual = labels.astype(bool)
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    tn = int(np.logical_and(~predicted, ~actual).sum())
    return tp, fp, fn, tn


def candidate_thresholds(scores: np.ndarray, grid_size: int = 200) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        raise ValueError("No finite scores available for threshold search.")
    lo = float(finite.min()) - 1e-6
    hi = float(finite.max()) + 1e-6
    grid = np.linspace(lo, hi, grid_size)
    return np.unique(np.concatenate([grid, finite]))


def fit_threshold(
    question: str,
    labelled_scores: np.ndarray,
    labels: np.ndarray,
    corpus_scores: np.ndarray | None = None,
    recall_target: float = 0.9,
    precision_target: float = 0.7,
    credible_level: float = 0.95,
    grid_size: int = 200,
) -> ThresholdResult:
    corpus = corpus_scores if corpus_scores is not None else labelled_scores
    best: ThresholdResult | None = None

    for threshold in candidate_thresholds(labelled_scores, grid_size=grid_size):
        tp, fp, fn, tn = confusion_at_threshold(labelled_scores, labels, float(threshold))
        recall_lower = beta_lower_bound(tp, fn, credible_level)
        precision_lower = beta_lower_bound(tp, fp, credible_level)
        if recall_lower < recall_target or precision_lower < precision_target:
            continue

        accept_count = int((corpus >= threshold).sum())
        selectivity = accept_count / max(len(corpus), 1)
        result = ThresholdResult(
            question=question,
            accept_threshold=float(threshold),
            reject_threshold=float(-abs(threshold)),
            recall_lower=recall_lower,
            precision_lower=precision_lower,
            recall_target=recall_target,
            precision_target=precision_target,
            credible_level=credible_level,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            labelled_examples=len(labelled_scores),
            corpus_examples=len(corpus),
            accept_count=accept_count,
            selectivity=selectivity,
        )
        if best is None or result.accept_count < best.accept_count:
            best = result
        elif best is not None and result.accept_count == best.accept_count and result.accept_threshold > best.accept_threshold:
            best = result

    if best is None:
        raise ValueError(
            "No threshold satisfied the Bayesian recall/precision lower-bound constraints. "
            "Lower targets, add labelled examples, or inspect score quality."
        )
    return best


def profile_dataframe(
    df: pd.DataFrame,
    scorer: OllamaLogOddsScorer,
    recall_target: float = 0.9,
    precision_target: float = 0.7,
    credible_level: float = 0.95,
    grid_size: int = 200,
) -> dict:
    required = {"review", "question", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    thresholds = []
    scored_rows = []

    for question, group in df.groupby("question", sort=False):
        scores = []
        labels = []
        for row in group.itertuples(index=False):
            score = scorer.score(str(row.review), str(question))
            label = parse_label(row.label)
            scores.append(score)
            labels.append(label)
            scored_rows.append({"question": question, "label": label, "log_odds": score})

        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(labels, dtype=int)
        thresholds.append(
            fit_threshold(
                question=str(question),
                labelled_scores=scores_arr,
                labels=labels_arr,
                recall_target=recall_target,
                precision_target=precision_target,
                credible_level=credible_level,
                grid_size=grid_size,
            ).to_json_dict()
        )

    return {
        "model": scorer.model,
        "api_base": scorer.api_base,
        "thresholds": thresholds,
        "scored_examples": scored_rows,
    }


def save_thresholds(profile: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(profile, indent=2), encoding="utf-8")
