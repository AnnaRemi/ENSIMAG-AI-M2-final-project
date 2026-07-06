# SUQL Movie Database

**Structured and Unstructured Query Language for IMDb reviews**

Inspired by the Stanford OVAL paper:  
[SUQL: Conversational Search over Structured and Unstructured Data with LLMs](https://arxiv.org/abs/2311.09818)  
GitHub source: https://github.com/stanford-oval/suql

---

## What This Project Does

The IMDb dataset has two kinds of data:

| Type | Columns | Query method |
|------|---------|-------------|
| Structured | `movie_id`, `title`, `year`, `runtime`, `director`, `genres` | Standard SQL predicates |
| Unstructured | `review` (free-text prose) | LLM-evaluated `answer()` operator |

SUQL lets you ask hybrid questions like:

> *"What are the top 5 movies released in 1999 considered amazing in reviews?"*

which it translates into:

```sql
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE year = 1999
  AND answer(review, 'Is this movie considered amazing in the review?') = 'Yes'
LIMIT 5;
```

The structured `WHERE year = 1999` part runs on SQLite (fast, free).  
The `answer(review, '...')` part calls an LLM only for rows that passed the structured filter (expensive, but cached).

---

## Architecture

```
Natural Language Question
        ↓  LLM few-shot semantic parser (in-context learning)
   SUQL Query
        ↓
   ┌────────────────────────────────────────────┐
   │              SUQL Compiler                 │
   │                                            │
   │  1. Strip answer()/summary() calls         │
   │  2. Execute structural SQL on SQLite  ←── fast (no LLM)
   │  3. For each surviving row:                │
   │       evaluate answer() via LLM  ←──────── expensive (cached)
   │  4. Compute summary() for output  ←──────── expensive (cached)
   └────────────────────────────────────────────┘
        ↓
   Results DataFrame  →  CSV
```

**Key optimisation from the paper (§5):** structured predicates run first, reducing the number of expensive LLM `answer()` calls. Results are cached by (review_hash, question) so repeated queries cost nothing.

---

## Files

```
suql_movies/
├── suql_engine.py      # Core engine: NL→SUQL parser, answer(), summary(), executor
├── ../scripts/run_suql.py
│                       # Shared CLI entry point + example questions
├── ../data/
│   └── imdb_joined.csv # IMDb dataset
└── outputs/            # CSV results written here by default
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

The project uses LiteLLM with an Ollama-compatible server. By default it connects
to `http://localhost:11434` and uses `ollama/gemma4:e4b`.

You can override these values:

```bash
export SUQL_API_BASE="http://localhost:11434"
export SUQL_MODEL="ollama/gemma4:e4b"
export SUQL_DATA_PATH="../data/imdb_joined.csv"
```

---

## Usage

### Python script

```bash
# Run all built-in example questions
python ../scripts/run_suql.py --engine-dir .

# Ask a single NL question
python ../scripts/run_suql.py --engine-dir . \
  "What horror movies from the 80s do reviewers find genuinely scary?"

# Run a raw SUQL query (skips the NL parser)
python ../scripts/run_suql.py --engine-dir . --suql "SELECT movie_id, title, year, genres, summary(review) AS review_summary
FROM movies
WHERE genres LIKE '%Sci-Fi%'
  AND year >= 2000
  AND answer(review, 'Does the reviewer praise the special effects?') = 'Yes'
LIMIT 5;"

# Suppress progress output
python ../scripts/run_suql.py --engine-dir . --quiet "best comedies of the 2000s"
```

### Aker / LIG GPU

On Aker, start Ollama with the GPU job script, then run the project in another
shell on the same allocated host:

```bash
export SUQL_API_BASE="http://127.0.0.1:11434"
export SUQL_MODEL="ollama/gemma4:e4b"
python ../scripts/run_suql.py --engine-dir . \
  "What horror movies from the 80s do reviewers find genuinely scary?"
```

### Python API

```python
from suql_engine import ask, ask_with_suql, nl_to_suql, answer_fn, summary_fn

# Full pipeline: NL → SUQL → CSV
df = ask(
    "Top 5 drama films from 1995 with emotionally moving reviews",
    output_csv="outputs/drama_1995.csv",
)

# Skip NL parser — write SUQL directly
df = ask_with_suql("""
    SELECT movie_id, title, year, director, genres, summary(review) AS review_summary
    FROM movies
    WHERE genres LIKE '%Crime%'
      AND answer(review, 'Does the reviewer praise the script or writing?') = 'Yes'
    LIMIT 10;
""")

# Use operators individually
print(answer_fn("This film is a towering masterpiece of cinema...", "Is the review positive?"))
# → 'Yes'

print(summary_fn("This film is a towering masterpiece..."))
# → 'The reviewer calls it a masterpiece of cinema.'
```

---

## SUQL Syntax Reference

### Structured predicates (standard SQL)

```sql
WHERE year = 1999
WHERE year BETWEEN 1990 AND 1999
WHERE runtime < 100
WHERE genres LIKE '%Horror%'
WHERE director LIKE '%Nolan%'
WHERE title LIKE '%Batman%'
```

### Free-text operator: `answer()`

```sql
-- Yes/No question
answer(review, 'Is this movie considered amazing?') = 'Yes'
answer(review, 'Does the reviewer dislike this film?') = 'No'

-- Factual question (returns a string — use with care in WHERE)
answer(review, 'What genre does the reviewer say this feels like?') = 'Comedy'
```

### Free-text operator: `summary()`

Used in SELECT only — generates a short prose summary of the review:

```sql
SELECT movie_id, title, summary(review) AS review_summary
```

### Combined example

```sql
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE genres LIKE '%Drama%'
  AND year >= 1990 AND year <= 1999
  AND answer(review, 'Is this film emotionally powerful or moving?') = 'Yes'
ORDER BY year DESC
LIMIT 5;
```

---

## Output Format

All results are saved as CSV files with these columns:

| Column | Description |
|--------|-------------|
| `movie_id` | IMDb identifier (e.g. `tt0111161`) |
| `title` | Movie title |
| `year` | Release year |
| `runtime` | Duration in minutes |
| `director` | Director name |
| `genres` | Comma-separated genre list |
| `review_summary` | LLM-generated summary of the review (when `summary()` is used) |

Raw `review` text is excluded from output when `summary()` is requested.

---

## Citation

```bibtex
@inproceedings{liu-etal-2024-suql,
    title = "{SUQL}: Conversational Search over Structured and Unstructured Data with Large Language Models",
    author = "Liu, Shicheng and Xu, Jialiang and Tjangnaka, Wesley and Semnani, Sina and Yu, Chen and Lam, Monica",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    year = "2024",
    url = "https://aclanthology.org/2024.findings-naacl.283",
}
```
