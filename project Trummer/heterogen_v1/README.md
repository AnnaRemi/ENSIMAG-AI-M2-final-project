# Multi-table heterogeneous retrieval

This directory contains `Heterogen_v1`, using the local SUQL IMDb data:

- movies: `project SUQL/data/imdb_structured_joined.csv`
- reviews: `project SUQL/data/imdb_reviews.csv`

The current question is:

> Which movies released in 1998 have reviews expressing a negative, critical, or strongly unfavorable opinion?

## Execution

1. Select structured movie rows from 1998.
2. Sample review rows.
3. Prefilter review candidates using exact `movie_id = tconst`.
4. Run a Trummer-style block join for the semantic sentiment predicate.
5. Write joined evidence, final movies, and per-call metrics.

The identifier prefilter prevents sending unrelated movie-review pairs to the LLM. If the LLM returns no pairs, the lightweight runner can add clearly labeled `deterministic_fallback` rows based on negative-review terms.

The pandas block-join operator requests schema-constrained JSON containing
stable `movie_id` values rather than positional `x,y` indexes. Returned IDs are
validated against the current movie and review blocks. Raw model responses and
parsed-pair counts are retained in the join statistics for diagnosis.

## Local setup

```bash
cd "/Users/annremizova/Desktop/lab m2/project Trummer/Heterogen_v1"
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run with Ollama:

```bash
python3 run_use_case3.py \
  --api-base http://127.0.0.1:11434 \
  --model ollama/gemma4:e4b
```

Dependency-light runner:

```bash
python3 run_use_case3_light.py \
  --data-dir "../../project SUQL/data" \
  --model gemma4:e4b \
  --max-movies 100 \
  --max-reviews 1000 \
  --prefilter-reviews-by-movie-id
```

## Aker

Sync this directory to:

```text
/home/daisy/remizova/project_Trummer/Heterogen_v1
```

Place the movie/review CSVs in:

```text
/home/daisy/remizova/project_Trummer/Heterogen_v1/data
```

Submit from Aker:

```bash
cd /home/daisy/remizova/project_Trummer/Heterogen_v1
bash scripts/run_aker_trummer_use_case3.sh
```

## Outputs

Runs are written under `outputs/aker_trummer_<timestamp>/` and contain:

- `use_case3_join_stats.csv`
- `use_case3_joined_pairs.csv`
- `use_case3_final_movies.csv`
- sibling `aker_trummer_<timestamp>.console.log`

Downloaded historical runs and OAR/Ollama logs are kept in `outputs/` and `logs/`.
