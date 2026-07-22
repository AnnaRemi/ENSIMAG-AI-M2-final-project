"""Validated loader for prompt-injection semantic dictionaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_CATEGORY_KEYS = {
    "template_type",
    "attribute",
    "seed_regex",
    "lexical_anchors",
    "few_shot_positive",
    "few_shot_hard_negative",
    "mining_stats",
    "provenance",
    "threshold",
}
ALLOWED_TEMPLATE_TYPES = {"recommend", "praise", "describe", "criticize"}


def validate_semantic_dict(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("semantic_dict must be a non-empty object keyed by category_id")
    for category_id, entry in value.items():
        if not isinstance(category_id, str) or not category_id:
            raise ValueError("every category_id must be a non-empty string")
        if not isinstance(entry, dict):
            raise ValueError(f"{category_id}: category entry must be an object")
        missing = REQUIRED_CATEGORY_KEYS - entry.keys()
        if missing:
            raise ValueError(f"{category_id}: missing keys: {sorted(missing)}")
        if entry["template_type"] not in ALLOWED_TEMPLATE_TYPES:
            raise ValueError(f"{category_id}: invalid template_type")
        if entry["threshold"] is not None:
            raise ValueError(f"{category_id}: mining output threshold must be null")
        anchors = entry["lexical_anchors"]
        if not isinstance(anchors, list) or any(
            not isinstance(anchor, dict)
            or not isinstance(anchor.get("term"), str)
            or not isinstance(anchor.get("z_score"), (int, float))
            for anchor in anchors
        ):
            raise ValueError(f"{category_id}: malformed lexical_anchors")
        for key in ("few_shot_positive", "few_shot_hard_negative"):
            if not isinstance(entry[key], list) or any(
                not isinstance(text, str) for text in entry[key]
            ):
                raise ValueError(f"{category_id}: {key} must be a list of strings")
        if not isinstance(entry["mining_stats"], dict):
            raise ValueError(f"{category_id}: mining_stats must be an object")
        if not isinstance(entry["provenance"], dict):
            raise ValueError(f"{category_id}: provenance must be an object")
    return value


def load_semantic_dict(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load UTF-8 JSON and validate the fields downstream prompt builders need."""
    with Path(path).open(encoding="utf-8") as handle:
        return validate_semantic_dict(json.load(handle))


__all__ = ["load_semantic_dict", "validate_semantic_dict"]
