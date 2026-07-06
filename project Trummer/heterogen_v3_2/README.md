# Trummer heterogen_v3_2: structured-pruned batched cascade

This implementation is the V3.2 Trummer variant used by `common_benchmark_10q`.
It is intentionally kept under `project Trummer` so the common benchmark only
loads and compares implementations instead of owning their logic.

Pipeline:

1. extract deterministic structured predicates from the question;
2. prune the movie table by those predicates;
3. prune reviews by the remaining `movie_id = tconst` keys;
4. score exact-ID candidates with the cheap model in batches;
5. learn the BARGAIN-style confidence cutoff from an expensive-model calibration sample;
6. accept or reject confident cheap decisions and send only uncertain candidates to the expensive model in larger batches.

Compared with `heterogen_v3`, V3.2 keeps the same structured prefilter but uses
the `heterogen_v2_3` batched cascade stage. Compared with `heterogen_v2_3`, V3.2
adds structured pruning before the batched cascade.

## Local

```bash
cd "/Users/annremizova/Desktop/lab m2/project Trummer/heterogen_v3_2"
python3 -m unittest discover -s tests -v
python3 run_use_case3.py --dry-run --output-dir outputs/local_dry_run
```

With a local Ollama server:

```bash
python3 run_use_case3.py \
  --api-base http://127.0.0.1:11434 \
  --cheap-model gemma4:e2b \
  --expensive-model gemma4:e4b \
  --question "Which movies released in 1998 have reviews expressing an overall negative, critical, or strongly unfavorable opinion of the movie?" \
  --output-dir outputs/local_llm
```

Each run writes `joined_evidence.csv`, `cascade_decisions.csv`,
`final_movies.csv`, and `run_metrics.json`. Metrics include cheap/expensive
calls and cheap/expensive time percentages.
