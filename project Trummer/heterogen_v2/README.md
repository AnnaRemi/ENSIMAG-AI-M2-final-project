# Trummer heterogen_v2: cascading semantic join

This implementation is separate from `heterogen_v1`. It performs:

1. an exact structured `movie_id = tconst` candidate join;
2. one cheap binary model score per candidate;
3. a BARGAIN-style calibration pass that asks the expensive model to label a
   sample and learns a confidence threshold from cheap/oracle agreement;
4. early acceptance or rejection only when the cheap score is confident enough;
5. batched expensive-model classification for the remaining candidates.

Cheap-model failures fail open to the expensive stage. The expensive prompt
contains explicit candidate pairs, so it cannot introduce unrelated
cross-product pairs. Every route and aggregate count is written to disk.
Run metrics also include `cheap_seconds`, `expensive_seconds`,
`cheap_time_percent`, and `expensive_time_percent`. Percentages use total
model-call time as their denominator.

## Local

```bash
cd "/Users/annremizova/Desktop/lab m2/project Trummer/heterogen_v2"
python3 -m unittest discover -s tests -v
python3 run_use_case3.py --dry-run --output-dir outputs/local_dry_run
```

With a local Ollama server:

```bash
python3 run_use_case3.py \
  --api-base http://127.0.0.1:11434 \
  --cheap-model gemma4:e2b \
  --expensive-model gemma4:e4b \
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
  "/Users/annremizova/Desktop/lab m2/project Trummer/heterogen_v2/" \
  remizova@aker.imag.fr:/home/daisy/remizova/project_Trummer/heterogen_v2/
rsync -az \
  "/Users/annremizova/Desktop/lab m2/common_benchmark_v2/data/" \
  remizova@aker.imag.fr:/home/daisy/remizova/project_Trummer/heterogen_v2/data/
```

Submit on Aker:

```bash
cd /home/daisy/remizova/project_Trummer/heterogen_v2
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
