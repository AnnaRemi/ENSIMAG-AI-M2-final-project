from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

try:
    from profiler import OllamaLogOddsScorer
except ImportError:  # pragma: no cover
    import sys

    STAGE_1_DIR = Path(__file__).resolve().parents[1] / "Stage_1"
    if str(STAGE_1_DIR) not in sys.path:
        sys.path.insert(0, str(STAGE_1_DIR))
    from profiler import OllamaLogOddsScorer


DEFAULT_API_BASE = os.environ.get("SUQL_API_BASE", "http://localhost:11434")
DEFAULT_CHEAP_MODEL = os.environ.get("SUQL_CHEAP_MODEL", "ollama/gemma2:2b")
DEFAULT_EXPENSIVE_MODEL = os.environ.get(
    "SUQL_EXPENSIVE_MODEL",
    os.environ.get("SUQL_MODEL", "ollama/phi4-mini"),
)
DEFAULT_CHEAP_ACCEPT_FLOOR = float(os.environ.get("SUQL_CHEAP_ACCEPT_FLOOR", "4.0"))
DEFAULT_CHEAP_MIN_DECISION_RATE = float(os.environ.get("SUQL_CHEAP_MIN_DECISION_RATE", "0.3"))
DEFAULT_CHEAP_MIN_PROBES = int(os.environ.get("SUQL_CHEAP_MIN_PROBES", "5"))


def parse_disabled_questions(value: str | None) -> set[str]:
    if not value:
        return set()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value.split("||")
    if isinstance(parsed, str):
        parsed = [parsed]
    return {str(item).strip() for item in parsed if str(item).strip()}


def litellm_model_name(model: str) -> str:
    """Return the model name expected by LiteLLM's Ollama provider."""
    return model if model.startswith("ollama/") else f"ollama/{model}"


def ollama_model_name(model: str) -> str:
    """Return the model name expected by Ollama's native API."""
    return model.removeprefix("ollama/")

ANSWER_SYSTEM = """You are evaluating a single movie review to answer a question about it.

Rules:
- If the question is a yes/no question, answer ONLY with 'Yes' or 'No' (no other text).
- If the question asks for a specific piece of information (e.g. a name, year, rating),
  answer with a brief string (a few words at most).
- Do not explain your reasoning. Output only the answer."""


@dataclass
class CascadeStats:
    cache_hits: int = 0
    cache_misses: int = 0
    cheap_score_calls: int = 0
    cheap_score_failures: int = 0
    cheap_early_accept: int = 0
    cheap_early_reject: int = 0
    cheap_skipped: int = 0
    cheap_disabled: int = 0
    expensive_full_calls: int = 0
    cheap_score_failure_reasons: dict[str, int] = field(default_factory=dict)
    model_usage_by_question: dict[str, dict[str, object]] = field(default_factory=dict)

    def usage_for(self, question: str, cheap_model: str, expensive_model: str) -> dict[str, object]:
        usage = self.model_usage_by_question.setdefault(
            question,
            {
                "cheap_model": cheap_model,
                "expensive_model": expensive_model,
                "cache_hits": 0,
                "cache_misses": 0,
                "cheap_score_calls": 0,
                "cheap_score_failures": 0,
                "cheap_early_accept": 0,
                "cheap_early_reject": 0,
                "cheap_skipped": 0,
                "cheap_disabled": 0,
                "no_model_early_reject": 0,
                "expensive_full_calls": 0,
                "cheap_accept_floor": None,
                "cheap_min_decision_rate": None,
                "cheap_min_probes": None,
                "cheap_skip_reason": "",
                "cheap_score_failure_reasons": {},
            },
        )
        usage["cheap_model"] = cheap_model
        usage["expensive_model"] = expensive_model
        return usage

    def record_cheap_score_failure(self, question: str, cheap_model: str, expensive_model: str, exc: Exception) -> None:
        reason = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        reason = reason[:500]
        self.cheap_score_failure_reasons[reason] = self.cheap_score_failure_reasons.get(reason, 0) + 1

        usage = self.usage_for(question, cheap_model, expensive_model)
        reasons = usage.setdefault("cheap_score_failure_reasons", {})
        if isinstance(reasons, dict):
            reasons[reason] = int(reasons.get(reason, 0)) + 1

    def to_json_dict(self) -> dict:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cheap_score_calls": self.cheap_score_calls,
            "cheap_score_failures": self.cheap_score_failures,
            "cheap_early_accept": self.cheap_early_accept,
            "cheap_early_reject": self.cheap_early_reject,
            "cheap_skipped": self.cheap_skipped,
            "cheap_disabled": self.cheap_disabled,
            "expensive_full_calls": self.expensive_full_calls,
            "cheap_score_failure_reasons": self.cheap_score_failure_reasons,
            "model_usage_by_question": self.model_usage_by_question,
        }


@dataclass(frozen=True)
class CascadeThreshold:
    cheap_accept_threshold: float
    cheap_reject_threshold: float
    labelled_examples: int = 0
    early_accept_count: int = 0
    early_reject_count: int = 0


@dataclass
class CascadeAnswerFilter:
    thresholds_path: str | Path
    cheap_model: str = DEFAULT_CHEAP_MODEL
    expensive_model: str = DEFAULT_EXPENSIVE_MODEL
    api_base: str = DEFAULT_API_BASE
    cheap_accept_floor: float = DEFAULT_CHEAP_ACCEPT_FLOOR
    cheap_min_decision_rate: float = DEFAULT_CHEAP_MIN_DECISION_RATE
    cheap_min_probes: int = DEFAULT_CHEAP_MIN_PROBES
    cheap_disabled_questions: set[str] | list[str] | tuple[str, ...] | None = None
    timeout: float = 120.0
    stats: CascadeStats = field(default_factory=CascadeStats)

    def __post_init__(self) -> None:
        self.api_base = self.api_base.rstrip("/")
        self.cheap_model = litellm_model_name(self.cheap_model)
        self.expensive_model = litellm_model_name(self.expensive_model)
        self.cheap_accept_floor = float(self.cheap_accept_floor)
        self.cheap_min_decision_rate = float(self.cheap_min_decision_rate)
        self.cheap_min_probes = int(self.cheap_min_probes)
        self.cheap_disabled_questions = {
            str(question).strip()
            for question in (self.cheap_disabled_questions or set())
            if str(question).strip()
        }
        self.cheap_ollama_model = ollama_model_name(self.cheap_model)
        self.expensive_ollama_model = ollama_model_name(self.expensive_model)
        self.cheap_scorer = OllamaLogOddsScorer(
            model=self.cheap_model,
            api_base=self.api_base,
            timeout=self.timeout,
        )
        self.thresholds = self._load_thresholds(Path(self.thresholds_path))
        self.cache: dict[str, str] = {}

    @staticmethod
    def _load_thresholds(path: Path) -> dict[str, CascadeThreshold]:
        if not path.exists():
            raise FileNotFoundError(f"Stage_2 thresholds file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded: dict[str, CascadeThreshold] = {}
        for item in payload.get("thresholds", []):
            loaded[str(item["question"])] = CascadeThreshold(
                cheap_accept_threshold=float(item.get("cheap_accept_threshold", math.inf)),
                cheap_reject_threshold=float(item.get("cheap_reject_threshold", -math.inf)),
                labelled_examples=int(item.get("labelled_examples", 0) or 0),
                early_accept_count=int(item.get("early_accept_count", 0) or 0),
                early_reject_count=int(item.get("early_reject_count", 0) or 0),
            )
        if not loaded:
            raise ValueError(f"No thresholds found in {path}")
        return loaded

    @staticmethod
    def _cache_key(review_text: str, question: str) -> str:
        raw = f"{review_text}|||{question}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _record_cheap_skip(self, usage: dict[str, object]) -> None:
        self.stats.cheap_skipped += 1
        usage["cheap_skipped"] = int(usage["cheap_skipped"]) + 1

    def _record_cheap_disabled(self, usage: dict[str, object]) -> None:
        self.stats.cheap_disabled += 1
        usage["cheap_disabled"] = int(usage["cheap_disabled"]) + 1
        usage["cheap_skip_reason"] = "disabled for this question"

    def _cheap_decision_rate(self, usage: dict[str, object]) -> float:
        calls = int(usage.get("cheap_score_calls", 0) or 0)
        if calls == 0:
            return 0.0
        decisions = int(usage.get("cheap_early_accept", 0) or 0) + int(usage.get("cheap_early_reject", 0) or 0)
        return decisions / calls

    def _cheap_should_be_skipped(
        self,
        threshold: CascadeThreshold,
        usage: dict[str, object],
    ) -> bool:
        if self.cheap_min_decision_rate <= 0 or self.cheap_min_probes <= 0:
            return False

        profile_decisions = threshold.early_accept_count + threshold.early_reject_count
        if threshold.labelled_examples >= self.cheap_min_probes:
            profile_rate = profile_decisions / max(threshold.labelled_examples, 1)
            if profile_rate < self.cheap_min_decision_rate:
                usage["cheap_skip_reason"] = (
                    f"profile decision rate {profile_rate:.3f} below {self.cheap_min_decision_rate:.3f}"
                )
                return True

        calls = int(usage.get("cheap_score_calls", 0) or 0)
        if calls >= self.cheap_min_probes:
            observed_rate = self._cheap_decision_rate(usage)
            if observed_rate < self.cheap_min_decision_rate:
                usage["cheap_skip_reason"] = (
                    f"observed decision rate {observed_rate:.3f} below {self.cheap_min_decision_rate:.3f} "
                    f"after {calls} probes"
                )
                return True
        return False

    def answer(self, review_text: str, question: str) -> str:
        usage = self.stats.usage_for(question, self.cheap_model, self.expensive_model)
        usage["cheap_accept_floor"] = self.cheap_accept_floor
        usage["cheap_min_decision_rate"] = self.cheap_min_decision_rate
        usage["cheap_min_probes"] = self.cheap_min_probes
        usage["cheap_disabled_for_question"] = question in self.cheap_disabled_questions
        key = self._cache_key(review_text, question)
        if key in self.cache:
            self.stats.cache_hits += 1
            usage["cache_hits"] = int(usage["cache_hits"]) + 1
            return self.cache[key]

        self.stats.cache_misses += 1
        usage["cache_misses"] = int(usage["cache_misses"]) + 1
        if not review_text or len(review_text.strip()) < 10:
            self.stats.cheap_early_reject += 1
            usage["cheap_early_reject"] = int(usage["cheap_early_reject"]) + 1
            usage["no_model_early_reject"] = int(usage["no_model_early_reject"]) + 1
            self.cache[key] = "No"
            return "No"

        threshold = self.thresholds.get(question)
        if threshold is None:
            result = self.expensive_answer(review_text, question)
            self.cache[key] = result
            return result

        if question in self.cheap_disabled_questions:
            self._record_cheap_disabled(usage)
            result = self.expensive_answer(review_text, question)
            self.cache[key] = result
            return result

        if self._cheap_should_be_skipped(threshold, usage):
            self._record_cheap_skip(usage)
            result = self.expensive_answer(review_text, question)
            self.cache[key] = result
            return result

        self.stats.cheap_score_calls += 1
        usage["cheap_score_calls"] = int(usage["cheap_score_calls"]) + 1
        try:
            score = self.cheap_scorer.score(review_text, question)
        except Exception as exc:
            self.stats.cheap_score_failures += 1
            usage["cheap_score_failures"] = int(usage["cheap_score_failures"]) + 1
            self.stats.record_cheap_score_failure(question, self.cheap_model, self.expensive_model, exc)
            result = self.expensive_answer(review_text, question)
            self.cache[key] = result
            return result

        effective_accept_threshold = max(threshold.cheap_accept_threshold, self.cheap_accept_floor)
        usage["effective_accept_threshold"] = effective_accept_threshold
        usage["cheap_reject_threshold"] = threshold.cheap_reject_threshold

        if score >= effective_accept_threshold:
            self.stats.cheap_early_accept += 1
            usage["cheap_early_accept"] = int(usage["cheap_early_accept"]) + 1
            self.cache[key] = "Yes"
            return "Yes"
        if score <= threshold.cheap_reject_threshold:
            self.stats.cheap_early_reject += 1
            usage["cheap_early_reject"] = int(usage["cheap_early_reject"]) + 1
            self.cache[key] = "No"
            return "No"

        result = self.expensive_answer(review_text, question)
        self.cache[key] = result
        return result

    def expensive_answer(self, review_text: str, question: str) -> str:
        usage = self.stats.usage_for(question, self.cheap_model, self.expensive_model)
        self.stats.expensive_full_calls += 1
        usage["expensive_full_calls"] = int(usage["expensive_full_calls"]) + 1
        user_prompt = f"Review:\n{review_text[:1500]}\n\nQuestion: {question}"
        payload = {
            "model": self.expensive_ollama_model,
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 20},
        }
        response = httpx.post(f"{self.api_base}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = extract_text(response.json()).strip()
        low = result.lower().strip().rstrip(".")
        if low in {"yes", "no"}:
            return low.capitalize()
        return result


def extract_text(payload: dict) -> str:
    if "response" in payload:
        return str(payload["response"])
    message = payload.get("message")
    if isinstance(message, dict):
        return str(message.get("content", ""))
    choices = payload.get("choices") or []
    if choices:
        choice = choices[0]
        if "text" in choice:
            return str(choice["text"])
        message = choice.get("message")
        if isinstance(message, dict):
            return str(message.get("content", ""))
    return ""
