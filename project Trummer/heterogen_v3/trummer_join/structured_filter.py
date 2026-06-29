from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


NUMERIC_COLUMNS = {"year", "runtime"}
GENRE_WORDS = {
    "action",
    "adventure",
    "animation",
    "biography",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "family",
    "fantasy",
    "history",
    "horror",
    "music",
    "musical",
    "mystery",
    "news",
    "romance",
    "sci-fi",
    "scifi",
    "sport",
    "thriller",
    "war",
    "western",
}
COLUMN_ALIASES = {
    "genre": "genres",
    "genres": "genres",
    "released": "year",
    "release year": "year",
    "year": "year",
    "runtime": "runtime",
    "duration": "runtime",
    "director": "director",
    "directed by": "director",
    "title": "title",
    "movie": "title",
    "movie id": "movie_id",
    "id": "movie_id",
}


@dataclass(frozen=True)
class StructuredFilter:
    column: str
    op: str
    value: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "column": self.column,
            "op": self.op,
            "value": self.value,
            "source": self.source,
        }


def extract_structured_filters(
    question: str,
    columns: Iterable[str],
) -> list[StructuredFilter]:
    """Extract deterministic movie-table predicates from a natural question.

    This is intentionally conservative: it only emits filters that can be
    applied to existing structured columns. The remaining semantic condition is
    left for the Trummer block prompt.
    """

    available = {str(column) for column in columns}
    lowered = " ".join(question.lower().split())
    filters: list[StructuredFilter] = []

    def add(column: str, op: str, value: str, source: str) -> None:
        if column not in available:
            return
        candidate = StructuredFilter(column, op, str(value), source)
        if candidate not in filters:
            filters.append(candidate)

    if "year" in available:
        for match in re.finditer(r"\b(18|19|20)\d{2}\b", lowered):
            add("year", "eq", match.group(0), match.group(0))
        for pattern, op in [
            (r"\b(?:after|since)\s+((?:18|19|20)\d{2})\b", "gt"),
            (r"\b(?:before|older than|pre)\s+((?:18|19|20)\d{2})\b", "lt"),
            (r"\b(?:from|since)\s+((?:18|19|20)\d{2})\s+(?:onward|onwards|or later)\b", "ge"),
        ]:
            for match in re.finditer(pattern, lowered):
                add("year", op, match.group(1), match.group(0))

    if "runtime" in available:
        runtime_patterns = [
            (r"\b(?:runtime|duration)\s*(<=|>=|<|>|=)\s*(\d+)\b", None),
            (r"\b(?:under|less than|shorter than)\s+(\d+)\s*(?:min|mins|minutes)?\b", "lt"),
            (r"\b(?:over|more than|longer than)\s+(\d+)\s*(?:min|mins|minutes)?\b", "gt"),
            (r"\b(?:at least)\s+(\d+)\s*(?:min|mins|minutes)?\b", "ge"),
            (r"\b(?:at most)\s+(\d+)\s*(?:min|mins|minutes)?\b", "le"),
        ]
        for pattern, fixed_op in runtime_patterns:
            for match in re.finditer(pattern, lowered):
                if fixed_op is None:
                    op = _symbol_to_op(match.group(1))
                    value = match.group(2)
                else:
                    op = fixed_op
                    value = match.group(1)
                add("runtime", op, value, match.group(0))

    if "genres" in available:
        for genre in GENRE_WORDS:
            if re.search(rf"\b{re.escape(genre)}\b", lowered):
                normalized = "Sci-Fi" if genre == "scifi" else genre.title()
                add("genres", "contains", normalized, genre)

    for alias, column in sorted(COLUMN_ALIASES.items(), key=lambda item: -len(item[0])):
        if column not in available:
            continue
        escaped = re.escape(alias)
        explicit_patterns = [
            rf"\b{escaped}\s*(=|:)\s*['\"]?([^,'\";]+)['\"]?",
            rf"\b{escaped}\s+(?:is|equals|contains)\s+['\"]?([^,'\";]+)['\"]?",
        ]
        for pattern in explicit_patterns:
            for match in re.finditer(pattern, lowered):
                if len(match.groups()) == 2:
                    op = "eq" if match.group(1) == "=" else "contains"
                    value = match.group(2)
                else:
                    op = "contains"
                    value = match.group(1)
                add(column, op, _clean_value(value), match.group(0))

    if "director" in available:
        for match in re.finditer(
            r"\b(?:directed by|director)\s+(.+?)(?:\s+with\s+|\s+and\s+|\s+from\s+|\s+in\s+|$)",
            lowered,
        ):
            add("director", "contains", _clean_value(match.group(1)), match.group(0))

    return filters


def apply_structured_filters(
    frame: pd.DataFrame,
    filters: Iterable[StructuredFilter],
) -> pd.DataFrame:
    result = frame.copy()
    for item in filters:
        if item.column not in result.columns:
            continue
        series = result[item.column]
        if item.column in NUMERIC_COLUMNS:
            left = pd.to_numeric(series, errors="coerce")
            right = float(item.value)
            if item.op == "eq":
                mask = left == right
            elif item.op == "lt":
                mask = left < right
            elif item.op == "le":
                mask = left <= right
            elif item.op == "gt":
                mask = left > right
            elif item.op == "ge":
                mask = left >= right
            else:
                mask = series.astype(str).str.contains(item.value, case=False, na=False)
        elif item.op == "eq":
            mask = series.astype(str).str.lower() == item.value.lower()
        else:
            mask = series.astype(str).str.contains(re.escape(item.value), case=False, na=False)
        result = result[mask].copy()
    return result


def semantic_predicate_from_question(question: str) -> str:
    lowered = question.lower()
    if _has_any_term(lowered, ("negative", "critical", "unfavorable", "bad", "poor")):
        return "the review expresses an overall negative, critical, or strongly unfavorable opinion of the movie"
    if _has_any_term(lowered, ("positive", "favorable", "praised", "praise", "liked")):
        return "the review expresses an overall positive or favorable opinion of the movie"
    return "the review satisfies the remaining natural-language condition in the user question"


def _has_any_term(text: str, terms: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _symbol_to_op(symbol: str) -> str:
    return {
        "=": "eq",
        "<": "lt",
        "<=": "le",
        ">": "gt",
        ">=": "ge",
    }[symbol]


def _clean_value(value: str) -> str:
    cleaned = value.strip(" .,:;\"'")
    cleaned = re.sub(r"\bmovies?\b", "", cleaned).strip()
    return " ".join(cleaned.split())
