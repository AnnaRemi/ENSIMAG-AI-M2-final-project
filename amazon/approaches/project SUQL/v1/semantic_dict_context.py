"""Load the mined Amazon semantic dictionary and render it as prompt guidance.

The Amazon dictionary has a different shape from the IMDb one, so this is a
sibling implementation rather than a copy:

* IMDb keys on ten broad *categories* matched by keyword markers
  (``praise_humor``, ``criticize_pacing``, ...), with ``lexical_anchors`` and
  ``few_shot_positive`` / ``few_shot_hard_negative``.
* Amazon keys on the question's own *concept* (``comfortable_dress``,
  ``tarnishing_necklace``, ...) under a top-level ``concepts`` object, with
  ``semantic_question`` / ``positive_examples`` / ``negative_examples``.

Because the concept id is not visible in the prompt, questions are matched to a
concept by their stored ``semantic_question`` text, falling back to token
overlap. Set ``SEMANTIC_CONCEPT`` to select one explicitly.

``SEMANTIC_DICT_PATH`` overrides discovery; ``SEMANTIC_DICT_REQUIRED=0`` allows
a missing dictionary to degrade to empty guidance instead of raising.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

DICTIONARY_FILENAME = "semantic_dictionary.json"
_STOPWORDS = {
    "do", "does", "the", "a", "an", "is", "are", "as", "it", "its", "this",
    "that", "they", "them", "say", "says", "said", "describe", "described",
    "describes", "reviews", "review", "reviewers", "product", "item", "which",
    "and", "or", "to", "of", "for", "in", "on", "with", "be", "been", "after",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", str(text).lower()) if t not in _STOPWORDS}


def _dictionary_path() -> Path | None:
    configured = os.environ.get("SEMANTIC_DICT_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.exists() else None
    for parent in Path(__file__).resolve().parents:
        for relative in (
            Path("semantic_dict") / DICTIONARY_FILENAME,
            Path("benchmarks") / "semantic_dict" / DICTIONARY_FILENAME,
        ):
            path = parent / relative
            if path.exists():
                return path
    return None


def _required() -> bool:
    return os.environ.get("SEMANTIC_DICT_REQUIRED", "1") != "0"


@lru_cache(maxsize=1)
def load_semantic_guidance() -> dict[str, dict]:
    path = _dictionary_path()
    if path is None:
        if _required():
            raise FileNotFoundError(
                f"Mined semantic dictionary ({DICTIONARY_FILENAME}) not found. "
                "Set SEMANTIC_DICT_PATH, or SEMANTIC_DICT_REQUIRED=0 to run without guidance."
            )
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        if _required():
            raise RuntimeError(f"Could not read semantic dictionary at {path}: {error}") from error
        return {}
    if not isinstance(value, dict):
        return {}
    concepts = value.get("concepts", value)
    return concepts if isinstance(concepts, dict) else {}


def category_for_question(question: str) -> str | None:
    """Resolve a question to a concept id via stored text, then token overlap."""
    explicit = os.environ.get("SEMANTIC_CONCEPT", "").strip()
    concepts = load_semantic_guidance()
    if explicit and explicit in concepts:
        return explicit
    asked = " ".join(str(question).lower().split())
    if not asked:
        return None
    for concept, entry in concepts.items():
        stored = " ".join(str(entry.get("semantic_question", "")).lower().split())
        if stored and (stored == asked or stored in asked or asked in stored):
            return concept
    asked_tokens = _tokens(asked)
    if not asked_tokens:
        return None
    best, best_score = None, 0.0
    for concept, entry in concepts.items():
        candidate = _tokens(entry.get("semantic_question", "")) | _tokens(concept.replace("_", " "))
        if not candidate:
            continue
        score = len(asked_tokens & candidate) / len(asked_tokens | candidate)
        if score > best_score:
            best, best_score = concept, score
    # Require a real overlap; an arbitrary nearest concept is worse than none.
    return best if best_score >= 0.20 else None


def semantic_guideline(question: str) -> str:
    """Render mined hints for the matching concept; never return a decision rule."""
    concept = category_for_question(question)
    entry = load_semantic_guidance().get(concept or "")
    if not isinstance(entry, dict):
        return ""
    positives = [str(t).strip() for t in entry.get("positive_examples", []) if str(t).strip()]
    negatives = [str(t).strip() for t in entry.get("negative_examples", []) if str(t).strip()]
    anchors = sorted(_tokens(concept.replace("_", " ")))
    lines = [
        "MINED SEMANTIC GUIDELINE (advisory prompt context, not a keyword rule):",
        f"Concept: {concept}",
        f"Predicate: {entry.get('semantic_question', '')}",
        "Use the examples to distinguish actual predicate support from topic-only mentions.",
        "Anchors suggest where to inspect, but an anchor alone never proves YES and its absence never proves NO.",
        "Concept anchors: " + (", ".join(anchors) if anchors else "(none)"),
        "Positive proxy exemplars:",
    ]
    lines.extend(f"  + {text}" for text in positives)
    lines.append("Hard-negative exemplars (topic mentioned without sufficient predicate support):")
    lines.extend(f"  - {text}" for text in negatives)
    lines.append("Judge only the current review and copy evidence from it; never treat an exemplar as current evidence.")
    return "\n".join(lines)


__all__ = ["category_for_question", "load_semantic_guidance", "semantic_guideline"]
