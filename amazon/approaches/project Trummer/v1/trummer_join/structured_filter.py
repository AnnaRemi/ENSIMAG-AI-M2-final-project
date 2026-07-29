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


# Amazon's structured columns are all free text; price is stored with a currency
# symbol so it is matched as text rather than compared numerically.
NUMERIC_COLUMNS: set[str] = set()
# The IMDb port keyed off a closed genre vocabulary. Amazon has no equivalent
# controlled vocabulary -- category is empty for every row -- so product-type
# words are matched against the title instead.
PRODUCT_TYPE_WORDS = {
    "dress", "shoe", "shirt", "jacket", "hat", "watch", "bag", "sock", "sandal",
    "costume", "necklace", "bracelet", "earring", "sunglass", "sweater", "hoodie",
    "wallet", "jean", "scarf", "swimsuit", "bikini", "legging", "skirt", "blouse",
    "coat", "vest", "glove", "backpack", "slipper", "purse", "pajama", "boot",
}
COLUMN_ALIASES = {
    "title": "title",
    "product": "title",
    "product name": "title",
    "brand": "brand",
    "made by": "brand",
    "category": "category",
    "price": "price",
    "product id": "product_id",
    "asin": "product_id",
    "id": "product_id",
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
You are a SUQL semantic parser for an Amazon product table. Convert the user's
question into one SUQL query.

Table: products
Columns:
  product_id TEXT
  title TEXT
  brand TEXT
  category TEXT
  price TEXT

There is also conceptual review text available only through:
  answer(review, '<yes/no question>') = 'Yes'

Rules:
1. Put every product-table condition in normal SQL over products.
2. Use answer(review, ...) only for conditions that require reading review text.
3. Do not use answer() for title, brand, category, price, or product_id.
4. Treat product-type words as structured title predicates when possible:
   dresses -> title LIKE '%dress%'
   running shoes -> title LIKE '%shoe%'
   leather wallets -> title LIKE '%wallet%' AND title LIKE '%leather%'
5. Use LIKE '%value%' for title, brand and category.
6. Select only product_id, title, brand, category, price.
7. brand, category and price are frequently empty; prefer title predicates.
8. Output only raw SQL, no markdown and no explanation.

Examples:
Question: dresses that reviewers call comfortable
SELECT product_id, title, brand, category, price
FROM products
WHERE title LIKE '%dress%'
  AND answer(review, 'Do the reviews describe the product as comfortable to wear?') = 'Yes';

Question: Nike shoes with reviews saying they fit true to size
SELECT product_id, title, brand, category, price
FROM products
WHERE title LIKE '%shoe%'
  AND brand LIKE '%Nike%'
  AND answer(review, 'Do the reviews say the product fits true to size?') = 'Yes';

Question: Which leggings do reviewers say are see-through?
SELECT product_id, title, brand, category, price
FROM products
WHERE title LIKE '%legging%'
  AND answer(review, 'Do the reviews say the item is sheer or see-through when stretched?') = 'Yes';
""").strip()


_ANSWER_RE = re.compile(
    r"answer\s*\(\s*(\w+)\s*,\s*(['\"])(.*?)\2\s*\)",
    re.IGNORECASE | re.DOTALL,
)


def prune_product_frame(
    frame: pd.DataFrame,
    question: str,
    *,
    api_base: str = "http://127.0.0.1:11434",
    parser_model: str | None = None,
    request_timeout: float = 120.0,
    use_llm: bool = True,
    suql_query: str | None = None,
) -> tuple[pd.DataFrame, StructuredPruningResult]:
    """Prune product rows using SUQL-style structural parsing, then regex fallback.

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
        sqlite_frame.to_sql("products", conn, if_exists="replace", index=False)
        selected = pd.read_sql_query(structural_sql, conn)
    finally:
        conn.close()

    if "product_id" not in selected.columns:
        raise ValueError("structural SUQL must select product_id")
    selected_ids = [str(value) for value in selected["product_id"].tolist()]
    order = {product_id: index for index, product_id in enumerate(selected_ids)}
    pruned = frame[frame["product_id"].astype(str).isin(order)].copy()
    if not pruned.empty:
        pruned["_suql_order"] = pruned["product_id"].astype(str).map(order)
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
    """Extract deterministic product-table predicates from a natural question.

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

    if "price" in available:
        # price is stored as text with a currency symbol, so it is matched
        # literally rather than compared numerically.
        for match in re.finditer(r"\$\s?(\d+(?:\.\d{2})?)", lowered):
            add("price", "contains", match.group(1), match.group(0))

    if "title" in available:
        for word in PRODUCT_TYPE_WORDS:
            if re.search(rf"\b{re.escape(word)}s?\b", lowered):
                add("title", "contains", word, word)

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

    if "brand" in available:
        for match in re.finditer(
            r"\b(?:made by|brand|from)\s+(.+?)(?:\s+with\s+|\s+and\s+|\s+that\s+|\s+in\s+|$)",
            lowered,
        ):
            add("brand", "contains", _clean_value(match.group(1)), match.group(0))

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
        return "the review expresses an overall negative, critical, or strongly unfavorable opinion of the product"
    if _has_any_term(lowered, ("positive", "favorable", "praised", "praise", "liked")):
        return "the review expresses an overall positive or favorable opinion of the product"
    return "the review satisfies the remaining natural-language condition in the user question"


def semantic_predicate_from_suql(suql_query: str, question: str) -> str:
    answer_questions = [item[2] for item in _extract_answer_predicates(suql_query)]
    if answer_questions:
        if len(answer_questions) == 1:
            return answer_questions[0]
        return " and ".join(answer_questions)
    fallback = semantic_predicate_from_question(question)
    if fallback == "the review satisfies the remaining natural-language condition in the user question":
        return "the review is associated with the same product"
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
    cleaned = re.sub(r"\bproducts?\b", "", cleaned).strip()
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
    if not re.search(r"\bFROM\s+products\b", cleaned, flags=re.IGNORECASE):
        raise ValueError("structural SQL must read FROM products")
    forbidden = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA)\b"
    if re.search(forbidden, cleaned, flags=re.IGNORECASE):
        raise ValueError("structural SQL contains a forbidden statement")
    if ";" in cleaned.rstrip(";"):
        raise ValueError("structural SQL must contain one statement")
    return cleaned


def _sqlite_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("product_id", "title", "brand", "category", "price", "text"):
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
