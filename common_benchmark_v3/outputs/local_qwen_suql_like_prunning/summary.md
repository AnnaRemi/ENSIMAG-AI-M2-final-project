# All Heterogen versions: one-question comparison

Question: Which movies released in 1998 have reviews expressing an overall negative, critical, or strongly unfavorable opinion of the movie?

| label | wall_seconds | llm_calls | cheap_calls | expensive_calls | cheap_seconds | expensive_seconds | cheap_time_percent | expensive_time_percent | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2_2 | 6.7137 | 4.0000 | 0.0000 | 4.0000 | 0.0000 | 6.7036 | 0.0000 | 100.0000 | 0.8182 | 0.6923 | 0.7500 |
| V2_3 | 33.0059 | 10.0000 | 7.0000 | 3.0000 | 4.1009 | 28.9018 | 12.1184 | 87.8816 | 0.3333 | 0.9231 | 0.4898 |
| V3 | 11.6795 | 28.0000 | 25.0000 | 3.0000 | 7.3704 | 4.3049 | 66.8094 | 33.1906 | 0.5417 | 1.0000 | 0.7027 |

Stage percentages use cheap plus expensive model-call time as the denominator.
For V1 and V2_2 all LLM work is classified as expensive.
Token metrics are available for block-join implementations; unavailable metrics remain zero.

Generated plots:

- `metrics_precision_recall_f1.png`
- `time_bar_plot.png`
- `calls_bar_plot.png`
