# Cascade Threshold Sweep

Question: Which movies released in 1998 have reviews expressing an overall negative, critical, or strongly unfavorable opinion of the movie?

| label | threshold | precision | recall | f1 | final_answer_rows | cheap_early_accepts | cheap_early_rejects | expensive_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Row-wise cascade | 0.0000 | 0.2600 | 1.0000 | 0.4127 | 50.0000 | 50.0000 | 0.0000 | 0.0000 |
| Row-wise cascade | 0.5000 | 0.2600 | 1.0000 | 0.4127 | 50.0000 | 50.0000 | 0.0000 | 0.0000 |
| Row-wise cascade | 1.0000 | 0.2600 | 1.0000 | 0.4127 | 50.0000 | 50.0000 | 0.0000 | 0.0000 |
| Row-wise cascade | 1.5000 | 0.2600 | 1.0000 | 0.4127 | 50.0000 | 50.0000 | 0.0000 | 0.0000 |
| Row-wise cascade | 2.0000 | 0.2600 | 1.0000 | 0.4127 | 50.0000 | 50.0000 | 0.0000 | 0.0000 |
| Row-wise cascade | 2.5000 | 0.3250 | 1.0000 | 0.4906 | 40.0000 | 0.0000 | 0.0000 | 50.0000 |
| Row-wise cascade | 3.0000 | 0.3250 | 1.0000 | 0.4906 | 40.0000 | 0.0000 | 0.0000 | 50.0000 |
| Batch-wise cascade | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 16.0000 | 0.0000 | 7.0000 | 43.0000 |
| Batch-wise cascade | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 16.0000 | 0.0000 | 7.0000 | 43.0000 |
| Batch-wise cascade | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 16.0000 | 0.0000 | 7.0000 | 43.0000 |
| Batch-wise cascade | 1.5000 | 0.0000 | 0.0000 | 0.0000 | 16.0000 | 0.0000 | 7.0000 | 43.0000 |
| Batch-wise cascade | 2.0000 | 0.0000 | 0.0000 | 0.0000 | 16.0000 | 0.0000 | 7.0000 | 43.0000 |
| Batch-wise cascade | 2.5000 | 0.3333 | 0.9231 | 0.4898 | 36.0000 | 0.0000 | 0.0000 | 50.0000 |
| Batch-wise cascade | 3.0000 | 0.3333 | 0.9231 | 0.4898 | 36.0000 | 0.0000 | 0.0000 | 50.0000 |

Generated plots:

- `quality_metrics_vs_threshold.png`
- `final_rows_vs_threshold.png`
