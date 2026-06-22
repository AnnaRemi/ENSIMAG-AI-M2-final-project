from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

try:
    from profiler import OllamaLogOddsScorer
except ImportError:  # pragma: no cover - package-style import fallback
    from .profiler import OllamaLogOddsScorer


DEFAULT_MODEL = os.environ.get("SUQL_MODEL", "ollama/phi4-mini")
DEFAULT_API_BASE = os.environ.get("SUQL_API_BASE", "http://localhost:11434")

ANSWER_SYSTEM = """You are evaluating a single movie review to answer a question about it.

Rules:
- If the question is a yes/no question, answer ONLY with 'Yes' or 'No' (no other text).
- If the question asks for a specific piece of information (e.g. a name, year, rating),
  answer with a brief string (a few words at most).
- Do not explain your reasoning. Output only the answer."""


@dataclass
class FilterStats:
    cache_hits: int = 0
    cache_misses: int = 0
    llm_full_calls: int = 0
    llm_early_accept: int = 0
    llm_early_reject: int = 0
    llm_score_calls: int = 0
    llm_score_failures: int = 0

    def to_json_dict(self) -> dict:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "llm_full_calls": self.llm_full_calls,
            "llm_early_accept": self.llm_early_accept,
            "llm_early_reject": self.llm_early_reject,
            "llm_score_calls": self.llm_score_calls,
            "llm_score_failures": self.llm_score_failures,
        }


@dataclass(frozen=True)
class QuestionThreshold:
    accept_threshold: float
    reject_threshold: float


@dataclass
class CalibratedAnswerFilter:
    thresholds_path: str | Path
    model: str = DEFAULT_MODEL
    api_base: str = DEFAULT_API_BASE
    timeout: float = 120.0
    stats: FilterStats = field(default_factory=FilterStats)

    def __post_init__(self) -> None:
        self.api_base = self.api_base.rstrip("/")
        self.scorer = OllamaLogOddsScorer(model=self.model, api_base=self.api_base, timeout=self.timeout)
        self.thresholds = self._load_thresholds(Path(self.thresholds_path))
        self.cache: dict[str, str] = {}

    @staticmethod
    def _load_thresholds(path: Path) -> dict[str, QuestionThreshold]:
        if not path.exists():
            raise FileNotFoundError(f"Stage_1 thresholds file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded: dict[str, QuestionThreshold] = {}
        for item in payload.get("thresholds", []):
            loaded[str(item["question"])] = QuestionThreshold(
                accept_threshold=float(item["accept_threshold"]),
                reject_threshold=float(item["reject_threshold"]),
            )
        if not loaded:
            raise ValueError(f"No thresholds found in {path}")
        return loaded

    @staticmethod
    def _cache_key(review_text: str, question: str) -> str:
        raw = f"{review_text}|||{question}"
        return hashlib.md5(raw.encode()).hexdigest()

    def answer(self, review_text: str, question: str) -> str:
        key = self._cache_key(review_text, question)
        if key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[key]

        self.stats.cache_misses += 1
        threshold = self.thresholds.get(question)
        if threshold is None:
            result = self.full_answer(review_text, question)
            self.cache[key] = result
            return result

        if not review_text or len(review_text.strip()) < 10:
            self.stats.llm_early_reject += 1
            self.cache[key] = "No"
            return "No"

        self.stats.llm_score_calls += 1
        try:
            log_odds = self.scorer.score(review_text, question)
        except Exception:
            self.stats.llm_score_failures += 1
            result = self.full_answer(review_text, question)
            self.cache[key] = result
            return result

        if log_odds >= threshold.accept_threshold:
            self.stats.llm_early_accept += 1
            self.cache[key] = "Yes"
            return "Yes"
        if log_odds <= threshold.reject_threshold:
            self.stats.llm_early_reject += 1
            self.cache[key] = "No"
            return "No"

        result = self.full_answer(review_text, question)
        self.cache[key] = result
        return result

    def full_answer(self, review_text: str, question: str) -> str:
        self.stats.llm_full_calls += 1
        if not review_text or len(review_text.strip()) < 10:
            return "No"
        user_prompt = f"Review:\n{review_text[:1500]}\n\nQuestion: {question}"
        native_payload = {
            "model": self.model.removeprefix("ollama/"),
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 20},
        }
        response = httpx.post(
            f"{self.api_base}/api/chat",
            json=native_payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = extract_text(response.json()).strip()
        low = result.lower().strip().rstrip(".")
        if low in {"yes", "no"}:
            return low.capitalize()
        return result


def extract_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if choices:
        choice = choices[0]
        if "text" in choice:
            return str(choice["text"])
        message = choice.get("message")
        if isinstance(message, dict):
            return str(message.get("content", ""))
    if "response" in payload:
        return str(payload["response"])
    message = payload.get("message")
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return ""
