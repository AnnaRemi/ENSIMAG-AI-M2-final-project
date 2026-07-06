# Qwen3 V2_3 vs V3 Three-Question Comparison

Models:
- Cheap model: `ollama/qwen3:0.6b`
- Expensive model: `ollama/qwen3:1.7b`
- Cascade target: `0.9`
- Calibration budget: `20`
- Structured pruning mode: deterministic regex parser (`--disable-llm-structured-parser`)

## Aggregate

| Method | Total wall time | Total LLM calls | Cheap calls | Expensive calls | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Batch cascade V2_3 | 48.93 s | 16 | 10 | 6 | 0.406 | 0.733 | 0.491 |
| Pruned cascade V3 | 64.73 s | 85 | 70 | 15 | 0.462 | 0.867 | 0.589 |

## Plots

- `metrics_precision_recall_f1.png`: v3-style grouped bar plot for precision, recall, and F1. X-axis labels include the method and threshold source; Q1, Q2, and Q3 are shown at the top of the separated question boxes. The right-side legends include both the full question text and method context: V2_3 is batch-wise cascading, while V3 is structured-pruning plus row-wise cascading. `no learned / T=0.9` means no routing threshold was learned, so the cascade used the learned-threshold hyperparameter target `0.9` and fell back to expensive routing.
- `time_bar_plot.png`: v3-style stacked cheap/expensive model-call time plot with percentages inside bars and the same question/method-context legends.
- `calls_bar_plot.png`: v3-style stacked cheap/expensive call-count plot with segment counts, totals, and the same question/method-context legends.
- `metrics_by_question_with_thresholds.png`: copy of the v3-style metrics plot kept for the previous filename.
- `comparison_v2_3_vs_v3.png`: earlier compact overview plot for F1, wall time, calls, and final rows by question.
- `thresholds_used_by_question.csv`: threshold values used per question and method.

## Interpretation

V2_3 is faster and uses far fewer calls because it batches cheap scoring and expensive fallback. V3 has higher macro F1 because it returns more true positives on the medium question, but it pays for that with many more row-wise cheap calls.

The learned threshold was `null` for V2_3 on all three questions. V3 learned threshold `2.0` only on the hard question; on easy and medium it also fell back because calibration agreement did not reach the target.

A first run with the qwen3 LLM structured parser enabled pruned questions 2 and 3 to zero candidates because the parser produced incorrect structural filters. This summary uses the deterministic structured parser so the cascade comparison is meaningful.
