# Heterogen Structural/Cascade Aggregate Plots

Source: `common_benchmark_10q/outputs/gemma4_e2b_e4b_10q_11reps_20260705_184848/comparison.csv`

SUQL baseline is excluded. Values are averages across the 10 per-question result rows; each per-question row is the mean of 11 repetitions.

## Plots

- `01_quality_precision_recall_f1.png`
- `02_time_cheap_expensive_percent.png` - actual time bar height with cheap/expensive percentage slices.
- `03_calls_cheap_expensive_percent.png` - actual call-count bar height with cheap/expensive percentage slices.
- `04_quality_time_calls_tradeoff.png`

## Aggregate Metrics

| method | precision | recall | f1 | wall_seconds | llm_calls | cheap_calls | expensive_calls | cheap_seconds | expensive_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2_3 | 0.3559 | 0.4871 | 0.4030 | 13.8040 | 10.7091 | 8.0000 | 2.7091 | 10.2845 | 3.5182 |
| V3 | 0.6926 | 0.2136 | 0.3223 | 29.6639 | 30.0000 | 24.0000 | 6.0000 | 22.5985 | 7.0636 |
| V3_2 | 0.9900 | 0.4939 | 0.6069 | 5.9477 | 4.6909 | 3.0000 | 1.6909 | 3.7814 | 2.1656 |
