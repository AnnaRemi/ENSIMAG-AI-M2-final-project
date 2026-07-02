# All Heterogen versions: one-question comparison

Question: Which movies released in 1998 have reviews expressing an overall negative, critical, or strongly unfavorable opinion of the movie?

| label | wall_seconds | llm_calls | cheap_calls | expensive_calls | cheap_seconds | expensive_seconds | cheap_time_percent | expensive_time_percent | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | 19.1587 | 14.0000 | 0.0000 | 14.0000 | 0.0000 | 19.1442 | 0.0000 | 100.0000 | 0.2500 | 0.0769 | 0.1176 |
| V2 | 25.6460 | 60.0000 | 50.0000 | 10.0000 | 14.4065 | 11.2332 | 57.8829 | 42.1171 | 0.1600 | 0.3077 | 0.2105 |
| V2_3 | 5.6364 | 10.0000 | 7.0000 | 3.0000 | 2.9597 | 2.6731 | 57.9499 | 42.0501 | 0.0000 | 0.0000 | 0.0000 |

Stage percentages use cheap plus expensive model-call time as the denominator.
For V1 and V2_2 all LLM work is classified as expensive.
Token metrics are available for block-join implementations; unavailable metrics remain zero.

Generated plots:

- `metrics_precision_recall_f1.png`
- `time_bar_plot.png`
- `calls_bar_plot.png`
