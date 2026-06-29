# All Heterogen versions: one-question comparison

Question: Which movies released in 1998 have reviews expressing an overall negative, critical, or strongly unfavorable opinion of the movie?

| label | wall_seconds | llm_calls | cheap_calls | expensive_calls | cheap_seconds | expensive_seconds | cheap_time_percent | expensive_time_percent | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | 533.9821 | 14 | 0 | 14 | 0.0000 | 529.9218 | 0.0000 | 100.0000 | 0.8000 | 0.3077 | 0.4444 |
| V2 | 426.1414 | 54 | 50 | 4 | 388.8410 | 37.2994 | 91.2471 | 8.7529 | 0.0000 | 0.0000 | 0.0000 |
| V2_1 | 111.6330 | 14 | 0 | 14 | 0.0000 | 422.1401 | 0.0000 | 100.0000 | 0.8000 | 0.3077 | 0.4444 |
| V2_2 | 33.5527 | 4 | 0 | 4 | 0.0000 | 33.1083 | 0.0000 | 100.0000 | 1.0000 | 0.3077 | 0.4706 |
| V2_3 | 67.1716 | 8 | 7 | 1 | 49.1691 | 18.0020 | 73.1998 | 26.8002 | 1.0000 | 0.1538 | 0.2667 |
| V3 | 86.4134 | 27 | 25 | 2 | 62.9728 | 23.4401 | 72.8743 | 27.1257 | 0.0000 | 0.0000 | 0.0000 |

Stage percentages use cheap plus expensive model-call time as the denominator.
For V1, V2_1, and V2_2 all LLM work is classified as expensive.
For parallel V2_1, expensive_seconds is summed request time and may exceed wall time.
Token metrics are available for block-join implementations; unavailable metrics remain zero.

Generated plots:

- `metrics_precision_recall_f1.png`
- `time_bar_plot.png`
- `calls_bar_plot.png`
