# Trummer heterogen_v2: cascading semantic join

This implementation is separate from `heterogen_v1`. It performs:

1. an exact structured `movie_id = tconst` candidate join;
2. one cheap binary model score per candidate;
3. early acceptance or rejection outside the uncertainty band;
4. batched expensive-model classification only for uncertain candidates.

Cheap-model failures fail open to the expensive stage. The expensive prompt
contains explicit candidate pairs, so it cannot introduce unrelated
cross-product pairs. Every route and aggregate count is written to disk.

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
  --cheap-model gemma2:2b \
  --expensive-model qwen2.5:3b \
  --output-dir outputs/local_llm
```

The default thresholds are `reject <= -1.5` and `accept >= 3.0`. With Ollama's
hard `0/1` fallback scores (`-2/+2`), cheap negative decisions are rejected and
cheap positive decisions are verified by the expensive model. Calibrate these
thresholds on labeled data before treating them as final.

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
CHEAP_MODEL=gemma2:2b EXPENSIVE_MODEL=qwen2.5:3b PULL_MODELS=1 \
  bash scripts/run_gpu.sh
```

Inspect progress:

```bash
oarstat -u "$USER"
tail -F logs/cascade_*.console.log
```

Each run writes `joined_evidence.csv`, `cascade_decisions.csv`,
`final_movies.csv`, and `run_metrics.json` under `outputs/cascade_<timestamp>/`.
