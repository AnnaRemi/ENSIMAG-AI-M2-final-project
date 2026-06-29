# Trummer heterogen_v2_2: structured-pruned block join

This implementation combines SUQL-style deterministic filtering with the
Trummer block-join prompt strategy.

Execution plan:

1. extract structured movie-table predicates from the question;
2. apply those predicates to the structured movie table;
3. keep only reviews whose `tconst` appears in the selected `movie_id` set;
4. run a Trummer-style block join over the remaining movie and review rows;
5. ask the LLM only for the semantic review condition.

The extractor is schema-aware: it only emits filters for columns that exist in
the movie table. For the current IMDb schema, it supports `movie_id`, `title`,
`director`, `year`, `runtime`, and `genres`, including common aliases such as
`genre`, `released`, and `duration`.

Examples:

- `2001 drama movies with positive reviews` -> `year = 2001`,
  `genres contains Drama`; LLM checks positive review sentiment.
- `runtime under 90 horror movies with negative reviews` -> `runtime < 90`,
  `genres contains Horror`; LLM checks negative review sentiment.
- `director contains nolan movies with positive reviews` -> structured
  director pruning; LLM checks positive review sentiment.

Compared with `heterogen_v1`, this moves `year=1998` and `movie_id=tconst`
out of the LLM prompt workload. Compared with `heterogen_v2`, it still uses
block prompts to search for matching rows rather than one cheap score per
exact candidate plus expensive fallback.

## Local

```bash
cd "/Users/annremizova/Desktop/lab m2/project Trummer/heterogen_v2_2"
python3 -m unittest discover -s tests -v
python3 run_use_case3.py --dry-run --output-dir outputs/local_dry_run
```

With a local Ollama server:

```bash
python3 run_use_case3.py \
  --api-base http://127.0.0.1:11434 \
  --model ollama/gemma4:e4b \
  --question "2001 drama movies with positive reviews" \
  --output-dir outputs/local_llm
```
