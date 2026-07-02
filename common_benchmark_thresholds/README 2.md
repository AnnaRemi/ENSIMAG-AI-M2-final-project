# Common Benchmark Thresholds

## Description

`common_benchmark_thresholds/` is a threshold-sensitivity suite for the cascade
implementations from `common_benchmark_v3`. It does not introduce a new dataset:
it reuses the fixed 50-row v3 question and sweeps manual confidence thresholds
for two cascade granularities:

- V2 row-wise cascade, where each exact-ID candidate is scored separately.
- V2_3 batch-wise cascade, where cheap and expensive model requests classify
  batches of candidates.

Use this suite when the question is how threshold choice changes quality,
answer-set size, cheap early decisions, and expensive fallback load. Each
threshold/implementation pair is repeated by default, and the reported rows are
means over those repetitions.

Threshold sweep benchmark for the `common_benchmark_v3` cascade implementations.

This benchmark treats cascade confidence threshold as a hyperparameter. It runs
only:

- `v2`: row-wise cascade
- `v2_3`: batch-wise cascade

For each threshold, each implementation is run 9 times by default. Numeric
metrics and quality metrics are averaged across repetitions.

## Local Run

```bash
python3 common_benchmark_thresholds/scripts/run_threshold_sweep.py \
  --cheap-model ollama/gemma3:270m \
  --expensive-model ollama/gemma3:1b \
  --thresholds 0,0.5,1,1.5,2,2.5,3 \
  --repetitions 9
```

For Qwen:

```bash
python3 common_benchmark_thresholds/scripts/run_threshold_sweep.py \
  --cheap-model ollama/qwen3:0.6b \
  --expensive-model ollama/qwen3:1.7b \
  --thresholds 0,0.5,1,1.5,2,2.5,3 \
  --repetitions 9
```

## Outputs

Default output directory:

```text
common_benchmark_thresholds/outputs/threshold_sweep__cheap_<cheap>__expensive_<expensive>
```

Main artifacts:

- `threshold_metrics.csv`: one averaged row per implementation and threshold
- `summary.md`: compact table of averaged quality and routing metrics
- `quality_metrics_vs_threshold.png`: precision, recall, and F1 vs threshold
- `final_rows_vs_threshold.png`: final answer rows vs threshold
- `<implementation>/<threshold>/run_metrics_repetitions.csv`: per-repetition metrics

## Notes

Threshold is applied as `abs(score) >= threshold`.

With hard 0/1 cheap model outputs, scores are usually represented as `-2` or
`+2`, so thresholds up to `2` behave similarly, while thresholds above `2`
route all candidates to the expensive model.
