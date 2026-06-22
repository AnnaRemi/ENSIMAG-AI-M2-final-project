"""
suql_engine.py
==============
Core SUQL engine for the IMDb movie database.

SUQL (Structured and Unstructured Query Language) — Stanford OVAL paper:
  https://arxiv.org/abs/2311.09818

Architecture:
  1. NL question  →  SUQL query  (LLM semantic parser, few-shot in-context learning)
  2. SUQL query   →  split structured and semantic retrieval
  3. Run structured retrieval on data/imdb_structured_joined.csv and semantic
     answer(review, question) retrieval on data/imdb_reviews.csv separately
  4. Join the retrieved rows on movie_id/tconst and return as CSV

Key SUQL primitives implemented:
  • answer(free_text_field, question_string)  → 'Yes'|'No'|<string>
  • summary(free_text_field)                  → short human-readable summary

This version intentionally differs from src_baseline: structured retrieval and
semantic retrieval are independent online branches, and only their outputs are
joined.
"""

from __future__ import annotations

import re
import textwrap
import json
import hashlib
import sqlite3
import pandas as pd
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Optional
from litellm import completion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Same LiteLLM/Ollama calling style as test_table_reclamation.py.
# This expects an Ollama-compatible server to be running and reachable at API_BASE.
MODEL = os.environ.get("SUQL_MODEL", "ollama/phi4-mini")
API_BASE = os.environ.get("SUQL_API_BASE", "http://localhost:11434")


def _default_data_path(filename: str) -> str:
    local_data = os.path.join(os.path.dirname(__file__), "data", filename)
    repo_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", filename)
    return local_data if os.path.exists(local_data) else repo_data


STRUCTURED_DATA_PATH = os.environ.get(
    "SUQL_STRUCTURED_DATA_PATH",
    _default_data_path("imdb_structured_joined.csv"),
)
REVIEWS_DATA_PATH = os.environ.get(
    "SUQL_REVIEWS_DATA_PATH",
    _default_data_path("imdb_reviews.csv"),
)

# Table schema exposed to the LLM semantic parser (mirrors the paper's §4 approach)
TABLE_SCHEMA = """
Table: movies
Columns (structured / categorical):
  movie_id  TEXT        -- unique IMDb identifier, e.g. tt0111161
  title     TEXT        -- movie title
  year      INTEGER     -- release year (1895–2011)
  runtime   INTEGER     -- duration in minutes
  director  TEXT        -- director full name
  genres    TEXT        -- comma-separated genre list, e.g. 'Drama,Romance'

Column (unstructured / free text):
  review    TEXT        -- one user review per movie (plain English prose)

SUQL free-text functions available on the 'review' column:
  answer(review, '<yes/no or open question>')
      → 'Yes' | 'No' | short answer string derived from the review text
  summary(review)
      → a brief human-readable summary of the review text
"""

# ---------------------------------------------------------------------------
# Few-shot examples (in-context learning — Section 4 & Appendix A.2 of paper)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """
-- Example 1
-- Question: What are movies with a runtime under 90 minutes that have funny reviews?
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE runtime < 90
  AND answer(review, 'Is this review funny or does it mention comedy?') = 'Yes'
LIMIT 10;

-- Example 2
-- Question: Top 5 drama movies from the 1990s that reviewers consider masterpieces?
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE genres LIKE '%Drama%'
  AND year >= 1990 AND year <= 1999
  AND answer(review, 'Does the reviewer consider this movie a masterpiece or give it very high praise?') = 'Yes'
ORDER BY year DESC
LIMIT 5;

-- Example 3
-- Question: Find horror movies with terrifying reviews directed by someone named Wes
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE genres LIKE '%Horror%'
  AND director LIKE '%Wes%'
  AND answer(review, 'Does the reviewer find this movie terrifying or scary?') = 'Yes';

-- Example 4
-- Question: What is the longest movie that reviewers found boring?
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE answer(review, 'Does the reviewer find this movie boring or slow-paced?') = 'Yes'
ORDER BY runtime DESC
LIMIT 1;

-- Example 5
-- Question: List 5 comedy movies from 2000 onwards with surprisingly positive reviews
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE genres LIKE '%Comedy%'
  AND year >= 2000
  AND answer(review, 'Is the review surprisingly positive or enthusiastic about the movie?') = 'Yes'
LIMIT 5;
"""

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

class _QueryMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self.llm_prompt_count = 0
        self.modeled_execution_seconds: float | None = None

    def increment_llm_prompts(self) -> None:
        with self._lock:
            self.llm_prompt_count += 1

    def set_modeled_execution_seconds(self, seconds: float) -> None:
        with self._lock:
            self.modeled_execution_seconds = seconds


_query_metrics: ContextVar[_QueryMetrics | None] = ContextVar("query_metrics", default=None)


@contextmanager
def _query_metrics_scope(verbose: bool):
    """
    Track query-level metrics for logs.

    Nested scopes reuse the outer counter so ask() includes the parser prompt,
    while direct execute_suql() calls still get their own metrics.
    """
    outermost = _query_metrics.get() is None
    token = None
    start_time = None
    metrics = None
    if outermost:
        metrics = _QueryMetrics()
        token = _query_metrics.set(metrics)
        start_time = time.perf_counter()

    try:
        yield
    finally:
        if outermost:
            elapsed_seconds = time.perf_counter() - start_time
            prompts_sent = metrics.llm_prompt_count
            modeled_seconds = metrics.modeled_execution_seconds
            if verbose:
                print("\nExecution metrics:")
                if modeled_seconds is not None:
                    print(f"  Query execution time: {modeled_seconds:.2f} seconds")
                    print(f"  Actual wall-clock time: {elapsed_seconds:.2f} seconds")
                else:
                    print(f"  Query execution time: {elapsed_seconds:.2f} seconds")
                print(f"  LLM prompts sent: {prompts_sent}")
            _query_metrics.reset(token)


def _llm_call(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Single LLM call using LiteLLM, matching the calling style used in
    test_table_reclamation.py.

    The rest of the SUQL engine calls this one function, so replacing this
    backend automatically changes the model used by nl_to_suql(), answer_fn(),
    and summary_fn().
    """
    metrics = _query_metrics.get()
    if metrics is not None:
        metrics.increment_llm_prompts()

    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        api_base=API_BASE,
        timeout=3600,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Step 1: NL → SUQL  (semantic parser, few-shot in-context learning)
# ---------------------------------------------------------------------------

PARSER_SYSTEM = textwrap.dedent(f"""
You are a SUQL (Structured and Unstructured Query Language) semantic parser.
SUQL extends SQL with two free-text functions:
  • answer(free_text_column, 'question')  → 'Yes' | 'No' | short string
  • summary(free_text_column)             → a short prose summary

{TABLE_SCHEMA}

Rules:
1. Use plain SQL WHERE clauses for ALL structured predicates (year, runtime, director, genres, title, movie_id).
2. Use answer() ONLY for predicates that require understanding the review text.
3. Put structured predicates and answer() predicates together in the WHERE clause.
4. The SELECT list must include: movie_id, title, year, runtime, director, genres.
   Add summary(review) AS review_summary whenever you use answer() on the review.
5. Use LIKE '%value%' for partial text matches on genres and director.
6. For "top N" questions without a numeric ranking field, use LIMIT N.
7. Output ONLY the raw SQL query — no markdown fences, no explanation.

Few-shot examples:
{FEW_SHOT_EXAMPLES}
""").strip()


def nl_to_suql(question: str) -> str:
    """
    Translate a natural language question into a SUQL query using
    few-shot in-context learning (Section 4 of the SUQL paper).
    """
    user_prompt = f"-- Question: {question}\n"
    raw = _llm_call(PARSER_SYSTEM, user_prompt, max_tokens=512)
    # Strip any accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
    return _extract_first_sql_statement(raw)


# ---------------------------------------------------------------------------
# Step 2: Load split data and execute structured SQL via sqlite3
# ---------------------------------------------------------------------------


def _load_structured_dataframe() -> pd.DataFrame:
    df = pd.read_csv(STRUCTURED_DATA_PATH)
    df = df.dropna(subset=["movie_id"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0).astype(int)
    df["title"] = df["title"].fillna("")
    df["director"] = df["director"].fillna("")
    df["genres"] = df["genres"].fillna("")
    return df


def _load_reviews_dataframe() -> pd.DataFrame:
    df = pd.read_csv(REVIEWS_DATA_PATH)
    if "tconst" in df.columns and "movie_id" not in df.columns:
        df = df.rename(columns={"tconst": "movie_id"})
    df = df.dropna(subset=["movie_id"])
    df["review"] = df["review"].fillna("")
    return df[["movie_id", "review"]]


def _build_sqlite(df: pd.DataFrame) -> sqlite3.Connection:
    """Load the DataFrame into a temporary in-memory SQLite DB."""
    conn = sqlite3.connect(":memory:")
    df.to_sql("movies", conn, if_exists="replace", index=False)
    return conn


# ---------------------------------------------------------------------------
# Step 3: answer() / summary() — LLM-based free-text operators
# ---------------------------------------------------------------------------

# Simple in-process cache keyed on (review_hash, question) to avoid
# repeat LLM calls for the same (text, question) pair (paper §5.3).
_answer_cache: dict[str, str] = {}

ANSWER_SYSTEM = textwrap.dedent("""
You are evaluating a single movie review to answer a question about it.

Rules:
- If the question is a yes/no question, answer ONLY with 'Yes' or 'No' (no other text).
- If the question asks for a specific piece of information (e.g. a name, year, rating),
  answer with a brief string (a few words at most).
- Do not explain your reasoning. Output only the answer.
""").strip()

SUMMARY_SYSTEM = textwrap.dedent("""
Summarize the following movie review in 1-2 concise sentences.
Capture the reviewer's overall sentiment and the most notable point.
Output only the summary — no preamble.
""").strip()


def _cache_key(text: str, question: str) -> str:
    raw = f"{text}|||{question}"
    return hashlib.md5(raw.encode()).hexdigest()


def answer_fn(review_text: str, question: str) -> str:
    """
    SUQL answer() operator.
    Returns 'Yes', 'No', or a short answer string extracted from review_text.
    """
    key = _cache_key(review_text, question)
    if key in _answer_cache:
        return _answer_cache[key]

    if not review_text or len(review_text.strip()) < 10:
        result = "No"
    else:
        prompt = f"Review:\n{review_text[:1500]}\n\nQuestion: {question}"
        result = _llm_call(ANSWER_SYSTEM, prompt, max_tokens=20)
        # Normalise yes/no
        low = result.lower().strip().rstrip(".")
        if low in ("yes", "no"):
            result = low.capitalize()

    _answer_cache[key] = result
    return result


def summary_fn(review_text: str) -> str:
    """SUQL summary() operator — returns a short prose summary of the review."""
    if not review_text or len(review_text.strip()) < 10:
        return "(no review)"
    key = _cache_key(review_text, "__summary__")
    if key in _answer_cache:
        return _answer_cache[key]
    result = " ".join(_llm_call(SUMMARY_SYSTEM, review_text[:2000], max_tokens=80).split())
    _answer_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Step 4: SUQL query execution — parse, split, execute
# ---------------------------------------------------------------------------

# Regex to find answer() and summary() calls in the query
_ANSWER_RE = re.compile(
    r"answer\s*\(\s*(\w+)\s*,\s*(['\"])(.*?)\2\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_SUMMARY_SELECT_RE = re.compile(
    r"summary\s*\(\s*(\w+)\s*\)",
    re.IGNORECASE,
)


def _extract_first_sql_statement(text: str) -> str:
    """
    Keep only the first SELECT statement from an LLM response.
    Local models sometimes append notes or extra few-shot examples.
    """
    match = re.search(r"\bSELECT\b", text, flags=re.IGNORECASE)
    if not match:
        return text.strip()

    sql = text[match.start():]
    in_quote: str | None = None
    for i, char in enumerate(sql):
        if char in ("'", '"'):
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
        elif char == ";" and in_quote is None:
            return sql[: i + 1].strip()

    lines = []
    for line in sql.splitlines():
        if line.startswith(("###", "Note:", "-- Question:", "User:", "System:")):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_answer_predicates(sql: str) -> list[tuple[str, str, str]]:
    """
    Return list of (full_match, column, question) for every answer() call in the SQL.
    """
    return [(m.group(0), m.group(1), m.group(3)) for m in _ANSWER_RE.finditer(sql)]


def _strip_answer_predicates(sql: str) -> str:
    """
    Remove answer() calls from WHERE clause so the query can run on SQLite.
    Handles:  AND answer(...)='Yes'   /   answer(...) = 'Yes' AND   /   WHERE answer(...)='Yes'
    """
    # Remove  AND answer(...)='<val>'  or  answer(...)='<val>' AND
    cleaned = re.sub(
        r"\bAND\s+answer\s*\([^)]+\)\s*=\s*['\"][^'\"]*['\"]",
        "",
        sql,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"answer\s*\([^)]+\)\s*=\s*['\"][^'\"]*['\"\s]*AND\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove standalone  WHERE answer(...)='<val>'
    cleaned = re.sub(
        r"\bWHERE\s+answer\s*\([^)]+\)\s*=\s*['\"][^'\"]*['\"]",
        "WHERE 1=1",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _strip_summary_from_select(sql: str) -> str:
    """
    Replace  summary(col) AS alias  with  col  in the SELECT list
    so SQLite can run the structural query.  We'll compute summary later.
    """
    return re.sub(
        r"summary\s*\(\s*(\w+)\s*\)\s+AS\s+\w+",
        r"\1",
        sql,
        flags=re.IGNORECASE,
    )


def _strip_summary_order_by(sql: str) -> str:
    """
    Remove ORDER BY summary(col) clauses before running SQLite.
    summary() is evaluated after structural SQL, so SQLite cannot sort by it.
    """
    return re.sub(
        r"\bORDER\s+BY\s+summary\s*\(\s*\w+\s*\)\s*(?:ASC|DESC)?\s*(?=\bLIMIT\b|;|$)",
        "",
        sql,
        flags=re.IGNORECASE,
    )


def _strip_function_order_by(sql: str) -> str:
    """
    Remove ORDER BY clauses that call unsupported functions such as decade(year).
    SQLite can still sort by real columns, but SUQL functions are post-processed.
    """
    return re.sub(
        r"\bORDER\s+BY\s+\w+\s*\([^)]*\)\s*(?:ASC|DESC)?\s*(?=\bLIMIT\b|;|$)",
        "",
        sql,
        flags=re.IGNORECASE,
    )


def _get_limit(sql: str) -> Optional[int]:
    match = re.search(r"\bLIMIT\s+(\d+)\s*;?\s*$", sql, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _strip_limit(sql: str) -> str:
    return re.sub(r"\bLIMIT\s+\d+\s*;?\s*$", "", sql, flags=re.IGNORECASE).rstrip() + ";"


def _select_list_match(sql: str) -> Optional[re.Match[str]]:
    return re.search(r"^\s*SELECT\s+(.*?)\s+FROM\s+", sql, flags=re.IGNORECASE | re.DOTALL)


def _get_selected_output_names(sql: str) -> set[str]:
    """Return simple output column names/aliases from the SELECT list."""
    match = _select_list_match(sql)
    if not match:
        return set()

    names: set[str] = set()
    for item in match.group(1).split(","):
        item = item.strip()
        alias_match = re.search(r"\s+AS\s+(\w+)\s*$", item, flags=re.IGNORECASE)
        if alias_match:
            names.add(alias_match.group(1).lower())
            continue

        simple_col = item.split(".")[-1].strip()
        if re.fullmatch(r"\w+", simple_col):
            names.add(simple_col.lower())

    return names


def _ensure_columns_selected(sql: str, columns: set[str]) -> str:
    """
    Add internal columns needed by answer()/summary() to the SELECT list.
    They are removed from the final result if the user did not request them.
    """
    if not columns:
        return sql

    match = _select_list_match(sql)
    if not match:
        return sql

    select_list = match.group(1)
    if select_list.strip() == "*":
        return sql

    selected = _get_selected_output_names(sql)
    missing = [col for col in sorted(columns) if col.lower() not in selected]
    if not missing:
        return sql

    insert_at = match.end(1)
    return f"{sql[:insert_at]}, {', '.join(missing)}{sql[insert_at:]}"


def _get_answer_expected(sql: str) -> dict[tuple[str, str], str]:
    """
    Build a mapping  {(column, question): expected_value}  from answer() predicates.
    e.g.  answer(review, 'Is it amazing?') = 'Yes'  →  {('review', 'Is it amazing?'): 'Yes'}
    """
    result: dict[tuple[str, str], str] = {}
    pattern = re.compile(
        r"answer\s*\(\s*(\w+)\s*,\s*(['\"])(.*?)\2\s*\)\s*=\s*(['\"])(.*?)\4",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(sql):
        col, question, expected = m.group(1), m.group(3), m.group(5)
        result[(col, question)] = expected
    return result


def _get_summary_columns(sql: str) -> dict[str, str]:
    """
    Return {alias: column} for each  summary(col) AS alias  in SELECT.
    """
    result: dict[str, str] = {}
    pattern = re.compile(
        r"summary\s*\(\s*(\w+)\s*\)\s+AS\s+(\w+)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(sql):
        result[m.group(2)] = m.group(1)
    return result


STRUCTURED_COLUMNS = {"movie_id", "title", "year", "runtime", "director", "genres"}
UNSTRUCTURED_COLUMNS = {"review"}


def _strip_unstructured_columns_from_select(sql: str) -> str:
    """
    Remove review columns from the SQLite SELECT list.
    The online join implementation gets reviews from imdb_reviews.csv after
    structured retrieval completes.
    """
    match = _select_list_match(sql)
    if not match:
        return sql

    select_list = match.group(1)
    if select_list.strip() == "*":
        return sql

    kept_items: list[str] = []
    for item in select_list.split(","):
        stripped = item.strip()
        alias_match = re.search(r"\s+AS\s+(\w+)\s*$", stripped, flags=re.IGNORECASE)
        item_without_alias = re.sub(r"\s+AS\s+\w+\s*$", "", stripped, flags=re.IGNORECASE)
        simple_name = item_without_alias.split(".")[-1].strip().lower()
        alias_name = alias_match.group(1).lower() if alias_match else None
        if simple_name in UNSTRUCTURED_COLUMNS or alias_name in UNSTRUCTURED_COLUMNS:
            continue
        kept_items.append(stripped)

    if not kept_items:
        kept_items = ["movie_id"]

    return f"{sql[:match.start(1)]}{', '.join(kept_items)}{sql[match.end(1):]}"


def _run_structured_retrieval(structural_sql: str, verbose: bool) -> pd.DataFrame:
    structured_df = _load_structured_dataframe()
    conn = _build_sqlite(structured_df)
    try:
        result_df = pd.read_sql_query(structural_sql, conn)
    except Exception as exc:
        conn.close()
        raise RuntimeError(
            f"SQLite execution failed.\nQuery:\n{structural_sql}\nError: {exc}"
        ) from exc
    conn.close()

    if verbose:
        print(f"Structural filter → {len(result_df)} candidate rows")

    return result_df


def _run_semantic_retrieval(
    answer_predicates: dict[tuple[str, str], str],
    verbose: bool,
) -> pd.DataFrame:
    reviews_df = _load_reviews_dataframe()
    if not answer_predicates:
        return reviews_df

    if reviews_df.empty:
        return reviews_df

    keep_mask = pd.Series([True] * len(reviews_df), index=reviews_df.index)

    for (col, question), expected in answer_predicates.items():
        if col not in reviews_df.columns:
            if verbose:
                print(f"  [warn] Column '{col}' not found — skipping answer() predicate")
            continue

        active_indexes = list(reviews_df[keep_mask].index)
        total_active = len(active_indexes)
        if verbose:
            print(f"  answer({col}, '{question[:60]}...') = '{expected}' — evaluating {total_active} reviews…")

        for position, idx in enumerate(active_indexes, start=1):
            text = str(reviews_df.at[idx, col])
            result = answer_fn(text, question)
            if result.strip().lower() != expected.strip().lower():
                keep_mask.at[idx] = False
            if verbose and (position % 25 == 0 or position == total_active):
                kept_so_far = int(keep_mask.loc[active_indexes[:position]].sum())
                print(f"    checked {position}/{total_active} reviews, kept {kept_so_far}")

    result_df = reviews_df[keep_mask].copy()
    if verbose:
        print(f"  Semantic retrieval → {len(result_df)} review rows")
    return result_df


def _execute_suql_impl(
    suql_query: str,
    df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Execute a SUQL query and return a DataFrame of results.

    Pipeline:
      1. Split into structural SQL (for SQLite) + free-text predicates (answer/summary).
      2. Run structural retrieval on imdb_structured_joined.csv.
      3. Run semantic answer() retrieval on imdb_reviews.csv when needed.
      4. Join both retrieval outputs on movie_id.
      5. Compute summary() values for requested columns.
      6. Return final DataFrame.
    """
    if df is not None:
        raise ValueError("src_online_join loads split data files directly; pass no df.")

    suql_query = _extract_first_sql_statement(suql_query)

    if verbose:
        print(f"\n{'='*60}")
        print("SUQL Query:")
        print(suql_query)
        print(f"{'='*60}")

    # --- Parse answer() and summary() predicates ---
    answer_predicates = _get_answer_expected(suql_query)
    summary_cols = _get_summary_columns(suql_query)
    final_limit = _get_limit(suql_query) if answer_predicates else None
    requested_output_names = _get_selected_output_names(suql_query)
    internal_cols = {col for col, _question in answer_predicates}
    internal_cols.update(summary_cols.values())
    internal_only_cols = {
        col for col in internal_cols
        if col.lower() not in requested_output_names
    }
    needs_reviews = bool(answer_predicates or summary_cols or "review" in requested_output_names)
    join_only_cols = {"movie_id"} if needs_reviews and "movie_id" not in requested_output_names else set()

    # --- Strip free-text ops, run structural SQL on SQLite ---
    structural_sql = _strip_answer_predicates(suql_query)
    structural_sql = _strip_summary_from_select(structural_sql)
    structural_sql = _strip_unstructured_columns_from_select(structural_sql)
    structural_sql = _strip_summary_order_by(structural_sql)
    structural_sql = _strip_function_order_by(structural_sql)
    if final_limit is not None:
        structural_sql = _strip_limit(structural_sql)
    required_structured_cols = internal_cols.intersection(STRUCTURED_COLUMNS)
    if needs_reviews:
        required_structured_cols.add("movie_id")
    structural_sql = _ensure_columns_selected(structural_sql, required_structured_cols)

    semantic_df = None
    if answer_predicates:
        if verbose:
            print("Running structured and semantic retrieval separately")

        structured_start = time.perf_counter()
        candidate_df = _run_structured_retrieval(structural_sql, verbose=verbose)
        structured_seconds = time.perf_counter() - structured_start

        semantic_start = time.perf_counter()
        semantic_df = _run_semantic_retrieval(answer_predicates, verbose=verbose)
        semantic_seconds = time.perf_counter() - semantic_start

        modeled_seconds = max(structured_seconds, semantic_seconds)
        metrics = _query_metrics.get()
        if metrics is not None:
            metrics.set_modeled_execution_seconds(modeled_seconds)
        if verbose:
            print(f"Structured retrieval time → {structured_seconds:.2f} seconds")
            print(f"Semantic retrieval time → {semantic_seconds:.2f} seconds")
            print(f"Modeled online retrieval time → {modeled_seconds:.2f} seconds")
    else:
        structured_start = time.perf_counter()
        candidate_df = _run_structured_retrieval(structural_sql, verbose=verbose)
        structured_seconds = time.perf_counter() - structured_start
        metrics = _query_metrics.get()
        if metrics is not None:
            metrics.set_modeled_execution_seconds(structured_seconds)
        if verbose:
            print(f"Structured retrieval time → {structured_seconds:.2f} seconds")

    if candidate_df.empty:
        return candidate_df

    if needs_reviews:
        if semantic_df is None:
            semantic_df = _load_reviews_dataframe()
        candidate_df = candidate_df.merge(
            semantic_df,
            on="movie_id",
            how="inner",
            sort=False,
        )
        if verbose:
            print(f"Join on movie_id → {len(candidate_df)} rows")

    if final_limit is not None:
        candidate_df = candidate_df.head(final_limit)

    # --- Compute summary() columns ---
    for alias, col in summary_cols.items():
        if col in candidate_df.columns:
            if verbose:
                print(f"  Computing summary({col}) for {len(candidate_df)} rows…")
            candidate_df[alias] = candidate_df[col].apply(
                lambda t: summary_fn(str(t))
            )
        else:
            candidate_df[alias] = ""

    # Drop columns added only so answer()/summary() could run.
    drop_cols = internal_only_cols.union(join_only_cols)
    if drop_cols:
        candidate_df = candidate_df.drop(columns=list(drop_cols), errors="ignore")

    return candidate_df.reset_index(drop=True)


def execute_suql(
    suql_query: str,
    df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Execute a SUQL query and log query-level metrics when verbose."""
    with _query_metrics_scope(verbose):
        return _execute_suql_impl(suql_query, df=df, verbose=verbose)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _ask_impl(
    question: str,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline:
      NL question → SUQL query → execute → return DataFrame (and optionally save CSV).

    Parameters
    ----------
    question   : Natural language question about the movie database.
    output_csv : If provided, save results to this CSV file path.
    verbose    : Print progress information.

    Returns
    -------
    pd.DataFrame with the query results.
    """
    if verbose:
        print(f"\n{'#'*60}")
        print(f"Question: {question}")
        print(f"{'#'*60}")

    # Step 1: NL → SUQL
    suql_query = nl_to_suql(question)
    if verbose:
        print(f"\n[Semantic Parser] Generated SUQL:\n{suql_query}\n")

    # Step 2–5: Execute SUQL against split data files
    results = execute_suql(suql_query, verbose=verbose)

    if verbose:
        print(f"\nResults ({len(results)} rows):")
        print(results.to_string(index=False))

    # Save CSV
    if output_csv:
        results.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nSaved → {output_csv}")

    return results


def ask(
    question: str,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a natural-language query and log query-level metrics when verbose."""
    with _query_metrics_scope(verbose):
        return _ask_impl(question, output_csv=output_csv, verbose=verbose)


def _ask_with_suql_impl(
    suql_query: str,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Execute a manually-written SUQL query directly (skip NL parsing).

    Parameters
    ----------
    suql_query : A SUQL query string.
    output_csv : If provided, save results to this CSV file path.
    verbose    : Print progress information.

    Returns
    -------
    pd.DataFrame with the query results.
    """
    results = execute_suql(suql_query, verbose=verbose)

    if output_csv:
        results.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nSaved → {output_csv}")

    return results


def ask_with_suql(
    suql_query: str,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a raw SUQL query and log query-level metrics when verbose."""
    with _query_metrics_scope(verbose):
        return _ask_with_suql_impl(suql_query, output_csv=output_csv, verbose=verbose)
