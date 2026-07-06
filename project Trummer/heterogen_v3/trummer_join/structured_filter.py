from __future__ import annotations

import json
import re
import sqlite3
import textwrap
import urllib.error
import urllib.request
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


@dataclass(frozen=True)
class StructuredPruningResult:
    mode: str
    filters: list[StructuredFilter]
    suql_query: str = ""
    structural_sql: str = ""
    parser_model: str = ""
    parser_error: str = ""
    semantic_predicate: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "filters": [item.as_dict() for item in self.filters],
            "suql_query": self.suql_query,
            "structural_sql": self.structural_sql,
            "parser_model": self.parser_model,
            "parser_error": self.parser_error,
            "semantic_predicate": self.semantic_predicate,
        }


STRUCTURED_PARSER_SYSTEM = textwrap.dedent("""
You are a SUQL semantic parser for an IMDb movie table. Convert the user's
question into one SUQL query.

Table: movies
Columns:
  movie_id TEXT
  title TEXT
  year INTEGER
  runtime INTEGER
  director TEXT
  genres TEXT

There is also conceptual review text available only through:
  answer(review, '<yes/no question>') = 'Yes'

Rules:
1. Put every movie-table condition in normal SQL over movies.
2. Use answer(review, ...) only for conditions that require reading review text.
3. Do not use answer() for director, title, year, runtime, movie_id, or genres.
4. Treat genre-like words as structured genre predicates when possible:
   funny/comedic/comedy -> genres LIKE '%Comedy%'
   scary/terrifying -> genres LIKE '%Horror%'
   romantic -> genres LIKE '%Romance%'
   sci-fi/science fiction -> genres LIKE '%Sci-Fi%'
5. Use LIKE '%value%' for director, title, and genres.
6. Select only movie_id, title, year, runtime, director, genres.
7. Output only raw SQL, no markdown and no explanation.

Examples:
Question: movies directed by Christopher Nolan which are funny
SELECT movie_id, title, year, runtime, director, genres
FROM movies
WHERE director LIKE '%Christopher Nolan%'
  AND genres LIKE '%Comedy%';

Question: movies directed by Wes with reviews saying they are scary
SELECT movie_id, title, year, runtime, director, genres
FROM movies
WHERE director LIKE '%Wes%'
  AND answer(review, 'Does the review say the movie is scary or terrifying?') = 'Yes';

Question: Which movies released in 1998 have reviews expressing an overall negative opinion?
SELECT movie_id, title, year, runtime, director, genres
FROM movies
WHERE year = 1998
  AND answer(review, 'Does the review express an overall negative, critical, or strongly unfavorable opinion of the movie?') = 'Yes';
""").strip()


_ANSWER_RE = re.compile(
    r"answer\s*\(\s*(\w+)\s*,\s*(['\"])(.*?)\2\s*\)",
    re.IGNORECASE | re.DOTALL,
)


def prune_movie_frame(
    frame: pd.DataFrame,
    question: str,
    *,
    api_base: str = "http://127.0.0.1:11434",
    parser_model: str | None = None,
    request_timeout: float = 120.0,
    use_llm: bool = True,
    suql_query: str | None = None,
) -> tuple[pd.DataFrame, StructuredPruningResult]:
    """Prune movie rows using SUQL-style structural parsing, then regex fallback.

    The preferred path mirrors SUQL: a cheap model maps the natural-language
    question to SUQL, answer() predicates are removed, and the structural SQL
    runs in in-memory SQLite. If that parser path is unavailable, the previous
    conservative regex extractor is used.
    """

    filters = extract_structured_filters(question, frame.columns)
    model = parser_model or ""
    if suql_query or (use_llm and model):
        try:
            query = suql_query or nl_to_structural_suql(
                question,
                api_base=api_base,
                model=model,
                request_timeout=request_timeout,
            )
            pruned, structural_sql = apply_suql_structural_pruning(frame, query)
            return pruned, StructuredPruningResult(
                mode="suql_sqlite",
                filters=filters,
                suql_query=query,
                structural_sql=structural_sql,
                parser_model=model,
                semantic_predicate=semantic_predicate_from_suql(query, question),
            )
        except Exception as exc:
            pruned = apply_structured_filters(frame, filters).reset_index(drop=True)
            return pruned, StructuredPruningResult(
                mode="regex_fallback",
                filters=filters,
                parser_model=model,
                parser_error=str(exc),
                semantic_predicate=semantic_predicate_from_question(question),
            )

    pruned = apply_structured_filters(frame, filters).reset_index(drop=True)
    return pruned, StructuredPruningResult(
        mode="regex",
        filters=filters,
        semantic_predicate=semantic_predicate_from_question(question),
    )


def nl_to_structural_suql(
    question: str,
    *,
    api_base: str,
    model: str,
    request_timeout: float,
) -> str:
    payload = {
        "model": _plain_model(model),
        "messages": [
            {"role": "system", "content": STRUCTURED_PARSER_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 256},
    }
    response = _post_json(f"{api_base.rstrip('/')}/api/chat", payload, request_timeout)
    content = str(response.get("message", {}).get("content", ""))
    return _extract_first_sql_statement(content)


def apply_suql_structural_pruning(
    frame: pd.DataFrame,
    suql_query: str,
) -> tuple[pd.DataFrame, str]:
    sql = _extract_first_sql_statement(suql_query)
    answer_predicates = _extract_answer_predicates(sql)
    structural_sql = _strip_answer_predicates(sql)
    structural_sql = _strip_summary_from_select(structural_sql)
    structural_sql = _strip_function_order_by(structural_sql)
    if answer_predicates:
        structural_sql = _strip_limit(structural_sql)
    structural_sql = _validate_structural_sql(structural_sql)

    sqlite_frame = _sqlite_ready_frame(frame)
    conn = sqlite3.connect(":memory:")
    try:
        sqlite_frame.to_sql("movies", conn, if_exists="replace", index=False)
        selected = pd.read_sql_query(structural_sql, conn)
    finally:
        conn.close()

    if "movie_id" not in selected.columns:
        raise ValueError("structural SUQL must select movie_id")
    selected_ids = [str(value) for value in selected["movie_id"].tolist()]
    order = {movie_id: index for index, movie_id in enumerate(selected_ids)}
    pruned = frame[frame["movie_id"].astype(str).isin(order)].copy()
    if not pruned.empty:
        pruned["_suql_order"] = pruned["movie_id"].astype(str).map(order)
        pruned = (
            pruned.sort_values("_suql_order")
            .drop(columns=["_suql_order"])
            .reset_index(drop=True)
        )
    return pruned.reset_index(drop=True), structural_sql


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


def semantic_predicate_from_suql(suql_query: str, question: str) -> str:
    answer_questions = [item[2] for item in _extract_answer_predicates(suql_query)]
    if answer_questions:
        if len(answer_questions) == 1:
            return answer_questions[0]
        return " and ".join(answer_questions)
    fallback = semantic_predicate_from_question(question)
    if fallback == "the review satisfies the remaining natural-language condition in the user question":
        return "the review is associated with the same movie"
    return fallback


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


def _extract_answer_predicates(sql: str) -> list[tuple[str, str, str]]:
    return [(m.group(1), m.group(2), m.group(3)) for m in _ANSWER_RE.finditer(sql)]


def _extract_first_sql_statement(text: str) -> str:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\bSELECT\b", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("parser response did not contain SELECT")
    sql = text[match.start():]
    in_quote: str | None = None
    for index, char in enumerate(sql):
        if char in ("'", '"'):
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
        elif char == ";" and in_quote is None:
            return sql[: index + 1].strip()
    return sql.strip().rstrip(";") + ";"


def _strip_answer_predicates(sql: str) -> str:
    answer_cmp = (
        r"answer\s*\(\s*\w+\s*,\s*(['\"]).*?\1\s*\)\s*=\s*(['\"]).*?\2"
    )
    cleaned = re.sub(
        rf"\s+\bAND\s+{answer_cmp}",
        "",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        rf"{answer_cmp}\s+\bAND\s+",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        rf"\bWHERE\s+{answer_cmp}",
        "WHERE 1=1",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned


def _strip_summary_from_select(sql: str) -> str:
    return re.sub(
        r"summary\s*\(\s*(\w+)\s*\)\s+AS\s+\w+",
        r"\1",
        sql,
        flags=re.IGNORECASE,
    )


def _strip_function_order_by(sql: str) -> str:
    return re.sub(
        r"\bORDER\s+BY\s+\w+\s*\([^)]*\)\s*(?:ASC|DESC)?\s*(?=\bLIMIT\b|;|$)",
        "",
        sql,
        flags=re.IGNORECASE,
    )


def _strip_limit(sql: str) -> str:
    return (
        re.sub(r"\bLIMIT\s+\d+\s*;?\s*$", "", sql, flags=re.IGNORECASE)
        .strip()
        .rstrip(";")
        + ";"
    )


def _validate_structural_sql(sql: str) -> str:
    cleaned = sql.strip()
    if not cleaned.endswith(";"):
        cleaned += ";"
    if not re.match(r"^\s*SELECT\b", cleaned, flags=re.IGNORECASE):
        raise ValueError("only SELECT structural SQL is allowed")
    if not re.search(r"\bFROM\s+movies\b", cleaned, flags=re.IGNORECASE):
        raise ValueError("structural SQL must read FROM movies")
    forbidden = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA)\b"
    if re.search(forbidden, cleaned, flags=re.IGNORECASE):
        raise ValueError("structural SQL contains a forbidden statement")
    if ";" in cleaned.rstrip(";"):
        raise ValueError("structural SQL must contain one statement")
    return cleaned


def _sqlite_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("year", "runtime"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("movie_id", "title", "director", "genres", "text"):
        if column in result.columns:
            result[column] = result[column].fillna("")
    if "review" not in result.columns:
        result["review"] = ""
    return result


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
