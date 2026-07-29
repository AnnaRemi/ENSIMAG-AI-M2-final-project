"""Load the mined IMDb semantic dictionary and render it as prompt guidance.

Restored from the retired `fix/` tree. Two things changed against that version:

* Path resolution also searches ``benchmarks/semantic_dict/``. The dictionary
  used to sit at the repository root, so walking up parents found it; it now
  lives under ``<dataset>/benchmarks/semantic_dict/`` and the old walk missed it.
* ``load_semantic_guidance`` can raise instead of silently returning ``{}``.
  A missing dictionary previously degraded to an empty guideline with no error,
  which is indistinguishable from the layer not being wired in at all.

Set ``SEMANTIC_DICT_PATH`` to override discovery, or ``SEMANTIC_DICT_REQUIRED=0``
to allow the soft failure.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

DICTIONARY_FILENAME = "semantic_dict.json"

CATEGORY_MARKERS = [
    ("recommend_general", ("recommend", "worth watch", "worth see", "must-see", "must see")),
    ("praise_humor", ("funny", "hilarious", "laugh", "humor", "humour")),
    ("praise_fear", ("scary", "frightening", "terrifying", "creepy", "chilling")),
    ("praise_acting", ("acting", "cast", "performance")),
    ("praise_visuals", ("visual", "cinematography", "special effects", "imagery")),
    ("praise_chemistry", ("chemistry", "relationship", "love story", "romantic")),
    ("criticize_pacing", ("pacing", "paced", "slow", "boring", "dragging", "tedious")),
    ("praise_excitement", ("exciting", "thrilling", "gripping", "suspenseful", "tense")),
    ("praise_originality", ("original", "inventive", "fresh", "unpredictable", "creative")),
    ("praise_ending_twist", ("ending", "finale", "resolution", "plot twist", "twist")),
]


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
    return value if isinstance(value, dict) else {}


def category_for_question(question: str) -> str | None:
    lowered = " ".join(str(question).lower().split())
    for category_id, markers in CATEGORY_MARKERS:
        if any(marker in lowered for marker in markers):
            return category_id
    return None


def semantic_guideline(question: str) -> str:
    """Render mined hints for the matching predicate; never return a decision rule."""
    category_id = category_for_question(question)
    entry = load_semantic_guidance().get(category_id or "")
    if not isinstance(entry, dict):
        return ""
    anchors = [str(item.get("term", "")).strip() for item in entry.get("lexical_anchors", [])]
    anchors = [term for term in anchors if term]
    positives = [str(text).strip() for text in entry.get("few_shot_positive", []) if str(text).strip()]
    negatives = [str(text).strip() for text in entry.get("few_shot_hard_negative", []) if str(text).strip()]
    lines = [
        "MINED SEMANTIC GUIDELINE (advisory prompt context, not a keyword rule):",
        f"Category: {category_id}",
        f"Predicate type: {entry.get('template_type', '')}",
        f"Attribute: {entry.get('attribute', '')}",
        "Use the examples to distinguish actual predicate support from topic-only mentions.",
        "Anchors suggest where to inspect, but an anchor alone never proves YES and its absence never proves NO.",
        "Mined lexical anchors: " + (", ".join(anchors) if anchors else "(none)"),
        "Positive proxy exemplars:",
    ]
    lines.extend(f"  + {text}" for text in positives)
    lines.append("Hard-negative exemplars (topic mentioned without sufficient predicate support):")
    lines.extend(f"  - {text}" for text in negatives)
    lines.append("Judge only the current review and copy evidence from it; never treat an exemplar as current evidence.")
    return "\n".join(lines)


__all__ = ["category_for_question", "load_semantic_guidance", "semantic_guideline"]
