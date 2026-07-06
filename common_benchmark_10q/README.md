# Common Benchmark 10Q

Ten-question benchmark comparing:

- `suql_baseline`
- `heterogen_v2_3`
- `heterogen_v3`
- `heterogen_v3_2`

The benchmark keeps each implementation under its source project. In
particular, V3.2 lives in `project Trummer/heterogen_v3_2` and combines V3
structured pruning with the V2.3 batched cascade; the benchmark runner imports
that implementation instead of defining V3.2-specific logic inside the
benchmark.

Each question has 60 rows and a non-empty ground-truth set. Ground truth is the
intersection of deterministic structured filters and the IMDb source sentiment
split. Each question/method run defaults to 11 repetitions; numeric metrics and
quality metrics are averaged before the final comparison CSV and plots are
written.

## Build Datasets

Run from the repository root:

```bash
python3 common_benchmark_10q/scripts/build_datasets.py
python3 -m unittest discover -s common_benchmark_10q/tests -v
```

## Local Run

```bash
python3 common_benchmark_10q/scripts/run_all.py \
  --cheap-model ollama/gemma4:e2b \
  --expensive-model ollama/gemma4:e4b \
  --repetitions 11 \
  --output-dir outputs/local_gemma4_e2b_e4b_10q_11reps
```

The run writes:

- `comparison.csv`
- `aggregate.csv`
- `summary.md`
- `plots/01_quality_precision_recall_f1.png`
- `plots/02_time_cheap_expensive_percent.png`
- `plots/03_calls_cheap_expensive_percent.png`
- `plots/04_quality_time_calls_tradeoff.png`
- `question_*/plots/01_quality_precision_recall_f1.png`
- `question_*/plots/02_time_cheap_expensive_percent.png`
- `question_*/plots/03_calls_cheap_expensive_percent.png`
- `question_*/plots/04_quality_time_calls_tradeoff.png`

## Aker Run

Local Mac:

```bash
bash common_benchmark_10q/scripts/sync_common_benchmark_10q_to_aker.sh
```

Aker login node:

```bash
cd /home/daisy/remizova/common_benchmark_10q_workspace
PULL_MODELS=1 bash common_benchmark_10q/scripts/runner.sh submit-aker
```

The OAR worker requests one GPU and refuses to continue unless an NVIDIA GPU is
visible and Ollama appears in `nvidia-smi` for both `gemma4:e2b` and
`gemma4:e4b`.
