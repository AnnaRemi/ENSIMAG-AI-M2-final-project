# All Heterogen versions: one-question comparison

Question: Which movies released in 1998 have reviews expressing an overall negative, critical, or strongly unfavorable opinion of the movie?

| label | wall_seconds | llm_calls | cheap_calls | expensive_calls | cheap_seconds | expensive_seconds | cheap_time_percent | expensive_time_percent | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | 23.9989 | 14.0000 | 0.0000 | 14.0000 | 0.0000 | 23.9785 | 0.0000 | 100.0000 | 0.4000 | 0.1538 | 0.2222 |
| V2 | 33.5566 | 60.0000 | 50.0000 | 10.0000 | 16.4539 | 17.0932 | 52.8691 | 47.1309 | 0.3250 | 1.0000 | 0.4906 |
| V2_2 | 11.2340 | 4.0000 | 0.0000 | 4.0000 | 0.0000 | 11.2249 | 0.0000 | 100.0000 | 0.8000 | 0.6154 | 0.6957 |
| V2_3 | 37.6516 | 10.0000 | 7.0000 | 3.0000 | 5.0737 | 32.5743 | 11.7472 | 88.2528 | 0.3333 | 0.9231 | 0.4898 |
| V3 | 13.3391 | 28.0000 | 25.0000 | 3.0000 | 8.2514 | 5.0842 | 66.0974 | 33.9026 | 0.5220 | 1.0000 | 0.6859 |

Stage percentages use cheap plus expensive model-call time as the denominator.
For V1 and V2_2 all LLM work is classified as expensive.
Token metrics are available for block-join implementations; unavailable metrics remain zero.

Generated plots:

- `metrics_precision_recall_f1.png`
- `time_bar_plot.png`
- `calls_bar_plot.png`
