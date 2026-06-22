# Stage 1: Bayesian Threshold Calibration for SUQL `answer()`

This directory implements a calibrated early-decision layer for a SUQL-style
`answer(review, question) = 'Yes'` filter.

## Idea

For each `(review, question)` pair, the profiler asks the Ollama model for a
single binary classification token:

```text
1 = Yes
0 = No
```

It then extracts:

```text
log_odds = log p('1') - log p('0')
```

from the model logprobs response. A labelled calibration CSV provides gold
labels. For each candidate threshold, the profiler computes:

```text
TP, FP, FN, TN
recall posterior    = Beta(1 + TP, 1 + FN)
precision posterior = Beta(1 + TP, 1 + FP)
```

The credible lower bounds are computed with the lower-tail quantile
`1 - credible_level`. With the default `credible_level = 0.95`, this is the
5% posterior quantile. The selected threshold is the highest/lowest-selectivity
threshold whose recall and precision lower bounds meet the requested targets.

At runtime:

```text
score >= accept_threshold  -> early Yes
score <= reject_threshold  -> early No
otherwise                  -> full answer() generation
```

The rejection threshold is symmetric:

```text
reject_threshold = -abs(accept_threshold)
```

## Dependencies

Only these packages are used by the calibration/runtime code:

```bash
pip install httpx numpy scipy pandas
```

The benchmark plot additionally requires:

```bash
pip install matplotlib
```

## Environment

```bash
export SUQL_MODEL="ollama/phi4-mini"
export SUQL_API_BASE="http://127.0.0.1:11434"
```

The profiler uses Ollama's OpenAI-compatible completions endpoint:

```text
$SUQL_API_BASE/v1/completions
```

The selected Ollama server/model must return top-token logprobs containing
both tokens `1` and `0`. Calibration fails clearly if those logprobs are not
available.

## Calibrate

Input CSV columns:

```text
review,question,label
```

Labels may be `1/0`, `yes/no`, or `true/false`.

Run:

```bash
cd Stage_1
python calibrate.py labelled_sample.csv --output thresholds.json
```

Optional targets:

```bash
python calibrate.py labelled_sample.csv \
  --output thresholds.json \
  --recall-target 0.9 \
  --precision-target 0.7 \
  --credible-level 0.95
```

## Runtime Filter

`answer_filter.py` exposes `CalibratedAnswerFilter`. It loads
`thresholds.json`, applies early accept/reject decisions, performs full
generation only for ambiguous scores, and tracks:

```text
cache_hits
cache_misses
llm_score_calls
llm_full_calls
llm_early_accept
llm_early_reject
```

## Benchmark

`benchmark_stage1.py` compares:

```text
src_baseline
src_baseline_stage1
```

It writes a timestamped run under:

```text
Stage_1/benchmarks/
```

with:

```text
metrics.csv
comparison_plot.png
logs/
outputs/
metrics_sidecars/
```

## Aker Sample-Size Experiment

To compare baseline and Stage 1 on the prepared repository data samples from
100 to 1500 rows, submit:

```bash
bash scripts/run_aker_baseline_stage1_data_samples.sh
```

The default sizes are:

```text
100 200 500 1000 1500
```

Override them if needed:

```bash
SIZES="100 500 1500" bash scripts/run_aker_baseline_stage1_data_samples.sh
```

Each run is written under:

```text
Stage_1/benchmarks/aker_baseline_stage1_data_sample_<size>_<timestamp>/
```

After all sizes finish, the script writes the aggregate plot and CSV summaries
under:

```text
Stage_1/benchmarks/baseline_vs_stage1_data_samples_<timestamp>/
```

The main plot is:

```text
metrics_vs_sample_size.svg
```
