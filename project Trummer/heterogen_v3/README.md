# Trummer heterogen_v3: structured-pruned cascading semantic join

This implementation combines the two optimizations from the previous variants:

1. extract deterministic structured predicates from the user question;
2. prune the movie table by those predicates;
3. prune reviews by the remaining `movie_id = tconst` keys;
4. score each pruned exact-ID candidate with the cheap model;
5. learn the cascade confidence threshold from an expensive-model calibration
   sample;
6. send only untrusted candidates to the expensive model.

Cheap-model failures fail open to the expensive stage. The expensive prompt
contains explicit candidate pairs, so it cannot introduce unrelated
cross-product pairs. The cheap stage retains the original pair-level binary
scorer, while the expensive stage is coalesced into at most 4 calls by default.
Every route and aggregate count is written to disk.
Run metrics also include `cheap_seconds`, `expensive_seconds`,
`cheap_time_percent`, and `expensive_time_percent`. Percentages use total
model-call time as their denominator.

For example, `2001 year drama movies with positive reviews` becomes structured
filters `year = 2001` and `genres contains Drama`; only the remaining exact-ID
movie-review pairs are sent to the cascade, while the review sentiment stays in
the LLM predicate.

## Local

```bash
cd "/Users/annremizova/Desktop/lab m2/project Trummer/heterogen_v3"
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

There are no fixed accept/reject thresholds. The cheap model's score sign is the
proxy label and `abs(score)` is confidence; the run learns the confidence cutoff
from the calibration sample. Tune `--cascade-target` and
`--calibration-budget` to control the oracle-agreement target and calibration
cost, not the threshold itself.

## Aker GPU

Sync code and data from the local Mac:

```bash
rsync -az \
  --exclude outputs --exclude logs --exclude data --exclude __pycache__ \
  "/Users/annremizova/Desktop/lab m2/project Trummer/heterogen_v3/" \
  remizova@aker.imag.fr:/home/daisy/remizova/project_Trummer/heterogen_v3/
rsync -az \
  "/Users/annremizova/Desktop/lab m2/common_benchmark_v2/data/" \
  remizova@aker.imag.fr:/home/daisy/remizova/project_Trummer/heterogen_v3/data/
```

Submit on Aker:

```bash
cd /home/daisy/remizova/project_Trummer/heterogen_v3
bash scripts/run_gpu.sh
```

Override models or install missing models:

```bash
CHEAP_MODEL=gemma4:e2b EXPENSIVE_MODEL=gemma4:e4b PULL_MODELS=1 \
  bash scripts/run_gpu.sh
```

Inspect progress:

```bash
oarstat -u "$USER"
tail -F logs/cascade_*.console.log
```

Each run writes `joined_evidence.csv`, `cascade_decisions.csv`,
`final_movies.csv`, and `run_metrics.json` under `outputs/cascade_<timestamp>/`.
