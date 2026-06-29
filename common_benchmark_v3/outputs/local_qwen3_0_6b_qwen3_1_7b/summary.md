# All Heterogen versions: one-question comparison

Question: Which movies released in 1998 have reviews expressing an overall negative, critical, or strongly unfavorable opinion of the movie?

| label | wall_seconds | llm_calls | cheap_calls | expensive_calls | cheap_seconds | expensive_seconds | cheap_time_percent | expensive_time_percent | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | 19.6608 | 14.0000 | 0.0000 | 14.0000 | 0.0000 | 19.6423 | 0.0000 | 100.0000 | 0.4000 | 0.1538 | 0.2222 |
| V2 | 24.9843 | 60.0000 | 50.0000 | 10.0000 | 13.2780 | 11.6961 | 57.8204 | 42.1796 | 0.3250 | 1.0000 | 0.4906 |
| V2_3 | 27.0061 | 10.0000 | 7.0000 | 3.0000 | 4.0264 | 22.9763 | 13.9045 | 86.0955 | 0.3333 | 0.9231 | 0.4898 |

Stage percentages use cheap plus expensive model-call time as the denominator.
For V1 and V2_2 all LLM work is classified as expensive.
Token metrics are available for block-join implementations; unavailable metrics remain zero.

Generated plots:

- `metrics_precision_recall_f1.png`
- `time_bar_plot.png`
- `calls_bar_plot.png`
