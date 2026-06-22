# Common benchmark: SUQL baseline vs Trummer heterogen_v1

This benchmark executes the same annotated movie/review rows with:

- `project SUQL/src_baseline`
- `project Trummer/heterogen_v1`

The benchmark question is:

> Which movies released in 1998 have reviews expressing an overall negative, critical, or strongly unfavorable opinion of the movie?

The fixed dataset contains one review per unique movie: six negative and six positive movies from 1998, plus four negative-review movies from 1997 that test the structured year filter. The six satisfying 1998 movie IDs are stored in `benchmark.json`; row-level labels and rationales are in `data/annotations.csv`.

## Metrics

Each adapter records:

- client-process CPU time (`cpu_seconds`)
- engine wall time (`engine_seconds`)
- total LLM calls
- raw final answer rows
- unique found movie IDs
- precision, recall, and F1 against the annotated movie IDs

CPU time does not include CPU/GPU consumed by the external Ollama server.

## Run locally

Start Ollama and ensure the shared model exists:

```bash
ollama serve
ollama pull gemma2:2b
```

Then, from `/Users/annremizova/Desktop/lab m2`:

```bash
"project SUQL/.venv/bin/python" common_benchmark/scripts/run_all.py \
  --api-base http://127.0.0.1:11434 \
  --model ollama/gemma2:2b
```

To validate the complete data, adapter, evaluation, table, and plotting pipeline without an Ollama server:

```bash
"project SUQL/.venv/bin/python" common_benchmark/scripts/run_all.py --dry-run
```

Dry-run output is explicitly labeled and must not be reported as an LLM experiment.

## Run individual implementations

```bash
"project SUQL/.venv/bin/python" common_benchmark/scripts/run_suql_baseline.py \
  --api-base http://127.0.0.1:11434 \
  --model ollama/gemma2:2b
```

```bash
"project SUQL/.venv/bin/python" common_benchmark/scripts/run_trummer.py \
  --api-base http://127.0.0.1:11434 \
  --model ollama/gemma2:2b
```

Regenerate evaluation artifacts from existing run metrics:

```bash
"project SUQL/.venv/bin/python" common_benchmark/scripts/evaluate_and_plot.py \
  --outputs-dir common_benchmark/outputs/gemma2_2b
```

## Outputs

Each model has its own experiment directory:

- `outputs/<model_name>/suql_baseline/`
- `outputs/<model_name>/trummer_heterogen_v1/`

Cross-implementation artifacts are stored alongside them:

- `outputs/<model_name>/comparison.csv`
- `outputs/<model_name>/comparison.md`
- `outputs/<model_name>/movie_id_outcomes.csv`
- `outputs/<model_name>/time_comparison.png`
- `outputs/<model_name>/workload_quality_comparison.png`

For example, `ollama/llama3.2` writes to `outputs/llama3.2/`, while
`ollama/phi4-mini` writes to `outputs/phi4-mini/`.

The Trummer adapter deliberately calls the reusable `block_join` operator directly. It does not use `run_use_case3_light.py`'s deterministic fallback, because fallback rows would invalidate LLM precision and recall.

## Fully remote Aker batch execution

The Aker workflow uses a non-interactive OAR GPU job. After `oarsub` prints a
job ID, the terminal and SSH connection can be closed without stopping the
experiment.

### 1. Local Mac: sync the benchmark and minimal implementation source

```bash
cd "/Users/annremizova/Desktop/lab m2"

AKER_HOST=remizova@aker.imag.fr \
  bash common_benchmark/scripts/sync_common_benchmark_to_aker.sh
```

The default remote workspace is:

```text
/home/daisy/remizova/common_benchmark_workspace
```

### 2. Aker login node: submit a non-interactive job

```bash
ssh remizova@aker.imag.fr
cd /home/daisy/remizova/common_benchmark_workspace

MODELS="gemma2:2b qwen2.5:3b" \
PULL_MODELS=1 \
WALLTIME=04:00:00 \
bash common_benchmark/scripts/submit_aker_common_benchmark.sh
```

`MODELS` is a whitespace-separated list of Ollama model tags. Each model is
run sequentially against both implementations and stored under:

```text
common_benchmark/outputs/<model_name>/
```

Set `PULL_MODELS=1` when a model may not already exist in the Ollama cache.
Model availability and download size depend on the Ollama installation and
network access on Aker. Suitable examples include:

```text
gemma2:2b
llama3.2:1b
llama3.2:3b
qwen2.5:1.5b
qwen2.5:3b
phi4-mini
```

Do not choose a model that exceeds the allocated GPU memory.

### 3. Aker login node: monitor the batch job

```bash
oarstat -u "$USER"
oarstat -f -j <jobid>
```

Use the authoritative `stdout_file` and `stderr_file` paths printed by
`oarstat -f -j`. With the default workspace they are normally:

```bash
tail -F /home/daisy/remizova/common_benchmark_workspace/common_benchmark/logs/oar_<jobid>.out
tail -F /home/daisy/remizova/common_benchmark_workspace/common_benchmark/logs/oar_<jobid>.err
```

Per-model progress is also written to:

```text
common_benchmark/logs/<model_name>_<jobid>_<timestamp>.console.log
```

### 4. Local Mac: retrieve results

```bash
cd "/Users/annremizova/Desktop/lab m2"

AKER_HOST=remizova@aker.imag.fr \
  bash common_benchmark/scripts/pull_common_benchmark_from_aker.sh
```

The downloaded experiments merge into local
`common_benchmark/outputs/<model_name>/`. Aker logs are placed under
`common_benchmark/aker_logs/`.

### Plot every metric across all downloaded models

```bash
cd "/Users/annremizova/Desktop/lab m2"

MPLCONFIGDIR=common_benchmark/.mplconfig \
"project SUQL/.venv/bin/python" \
  common_benchmark/scripts/plot_models_vs_metrics.py
```

This writes one model-vs-metric PNG for every gathered numeric metric under:

```text
common_benchmark/outputs/model_metric_plots/
```

SUQL is shown as a red line and Trummer as a blue line.

Generate the cross-model analysis report:

```bash
"project SUQL/.venv/bin/python" \
  common_benchmark/scripts/analyze_model_results.py
```

The report is written to:

```text
common_benchmark/outputs/model_metric_plots/analysis.md
```

### Retry only one implementation

If one implementation completed before the job failed, preserve its existing
remote output and skip it in the replacement job. For example, rerun only
Trummer for `gemma2:2b`:

```bash
MODELS="gemma2:2b" \
PULL_MODELS=0 \
SKIP_SUQL=1 \
TRUMMER_REQUEST_TIMEOUT=3600 \
bash common_benchmark/scripts/submit_aker_common_benchmark.sh
```
