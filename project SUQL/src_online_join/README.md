# SUQL Movie Database - Online Join

**Parallel structured and semantic retrieval for IMDb reviews**

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

The structured `WHERE year = 1999` part runs on SQLite over
`data/imdb_structured_joined.csv`. In parallel, `answer(review, '...')` scans
`data/imdb_reviews.csv` with the LLM. The two retrieval outputs are joined only
after both branches finish.

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
   │  2. Structured branch: SQLite retrieval    │
   │     on imdb_structured_joined.csv          │
   │  3. Semantic branch: answer() retrieval    │
   │     on imdb_reviews.csv                    │
   │  4. Join branch outputs on movie_id/tconst │
   │  5. Compute summary() for output           │
   └────────────────────────────────────────────┘
        ↓
   Results DataFrame  →  CSV
```

This version is meant to contrast with `src_baseline`: it does not first reduce
the semantic search space with structured predicates. It runs both retrieval
branches independently and joins their results online. Results are cached by
`(review_hash, question)` inside the process.

---

## Files

```
src_online_join/
├── suql_engine.py      # Core engine: NL→SUQL parser, answer(), summary(), executor
├── main.py             # CLI entry point + example questions
├── requirements.txt
└── outputs/            # CSV results written here

../data/
├── imdb_structured_joined.csv # structured movie table
└── imdb_reviews.csv           # review table keyed by tconst
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project uses LiteLLM with an Ollama-compatible server. By default it connects
to `http://localhost:11434` and uses `ollama/phi4-mini`.

You can override these values:

```bash
export SUQL_API_BASE="http://localhost:11434"
export SUQL_MODEL="ollama/phi4-mini"
export SUQL_STRUCTURED_DATA_PATH="../data/imdb_structured_joined.csv"
export SUQL_REVIEWS_DATA_PATH="../data/imdb_reviews.csv"
```

---

## Usage

### Python script

```bash
# Run all built-in example questions
python main.py

# Ask a single NL question
python main.py "What horror movies from the 80s do reviewers find genuinely scary?"

# Run a raw SUQL query (skips the NL parser)
python main.py --suql "SELECT movie_id, title, year, genres, summary(review) AS review_summary
FROM movies
WHERE genres LIKE '%Sci-Fi%'
  AND year >= 2000
  AND answer(review, 'Does the reviewer praise the special effects?') = 'Yes'
LIMIT 5;"

# Suppress progress output
python main.py --quiet "best comedies of the 2000s"
```

### Aker / LIG GPU

On Aker, start Ollama with the GPU job script, then run the project in another
shell on the same allocated host:

```bash
export SUQL_API_BASE="http://127.0.0.1:11434"
export SUQL_MODEL="ollama/phi4-mini"
python main.py "What horror movies from the 80s do reviewers find genuinely scary?"
```

### Jupyter Notebook

```bash
jupyter notebook suql_movies.ipynb
```

The notebook walks through every step with explanations and lets you run custom questions interactively.

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
