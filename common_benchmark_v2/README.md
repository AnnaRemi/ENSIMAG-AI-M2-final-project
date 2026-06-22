# Common benchmark v2: 50 mixed-year rows

This benchmark keeps the same question and implementations as `common_benchmark`:

> Which movies released in 1998 have reviews expressing an overall negative,
> critical, or strongly unfavorable opinion of the movie?

## Key differences from v1

- Exactly 50 unique movies and 50 corresponding reviews.
- 25 movies are from 1998 and 25 are from other years.
- The other-year rows span multiple release years.
- Ground truth is regenerated as `year = 1998 AND negative IMDb source sentiment`.
- SUQL applies `year = 1998` as its structured filter.
- Trummer receives all 50 movies and all 50 reviews; `year = 1998` is part of
  the LLM join predicate together with `movie_id = tconst` and negative sentiment.
- Trummer defaults to a 4,096-token planning threshold and at most 256 output
  tokens per block. Movie and review blocks are capped at 25 and 8 rows,
  respectively. This produces 14 bounded requests for the current dataset and
  avoids hour-long single-block generations on slower Ollama models.

The source IMDb file preserves the original 50K test-split ordering: rows
`0..12499` are negative and rows `12500..24999` are positive. The generated
`data/annotations.csv` records the source row, sentiment label, year decision,
and ground-truth decision for every selected movie.

## Build and verify locally

From `/Users/annremizova/Desktop/lab m2`:

```bash
"project SUQL/.venv/bin/python" common_benchmark_v2/scripts/build_dataset.py

"project SUQL/.venv/bin/python" common_benchmark_v2/scripts/run_all.py \
  --model ollama/gemma2:2b \
  --dry-run
```

Dry-run validates dataset wiring and evaluation only. It is not an LLM result.

## Run one model locally

```bash
"project SUQL/.venv/bin/python" common_benchmark_v2/scripts/run_all.py \
  --api-base http://127.0.0.1:11434 \
  --model ollama/gemma2:2b
```

## Model pool

The v2 Aker scripts default to the same seven-model pool used by v1:

```text
gemma2:2b
llama3.2
llama3.2:1b
mistral:7b
phi4-mini
qwen2.5:3b
qwen2.5:7b
```

Override `MODELS` to run only a subset.

## Aker workflow

Local Mac:

```bash
cd "/Users/annremizova/Desktop/lab m2"
AKER_HOST=remizova@aker.imag.fr \
  bash common_benchmark_v2/scripts/sync_common_benchmark_to_aker.sh
```

Aker login node:

```bash
cd /home/daisy/remizova/common_benchmark_v2_workspace
PULL_MODELS=1 WALLTIME=12:00:00 \
  bash common_benchmark_v2/scripts/submit_aker_common_benchmark.sh
```

Local Mac, after completion:

```bash
cd "/Users/annremizova/Desktop/lab m2"
AKER_HOST=remizova@aker.imag.fr \
  bash common_benchmark_v2/scripts/pull_common_benchmark_from_aker.sh
```

Outputs are written under `common_benchmark_v2/outputs/<model>/`.

## Cross-model plots

```bash
MPLCONFIGDIR=common_benchmark_v2/.mplconfig \
"project SUQL/.venv/bin/python" \
  common_benchmark_v2/scripts/plot_models_vs_metrics.py

"project SUQL/.venv/bin/python" \
  common_benchmark_v2/scripts/analyze_model_results.py
```
