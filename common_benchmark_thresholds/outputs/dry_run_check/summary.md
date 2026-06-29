# Cascade Threshold Sweep

Question: Which movies released in 1998 have reviews expressing an overall negative, critical, or strongly unfavorable opinion of the movie?

| label | threshold | precision | recall | f1 | final_answer_rows | cheap_early_accepts | cheap_early_rejects | expensive_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Row-wise cascade | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 13.0000 | 50.0000 | 0.0000 | 0.0000 |
| Row-wise cascade | 3.0000 | 1.0000 | 1.0000 | 1.0000 | 13.0000 | 0.0000 | 0.0000 | 50.0000 |
| Batch-wise cascade | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 13.0000 | 50.0000 | 0.0000 | 0.0000 |
| Batch-wise cascade | 3.0000 | 1.0000 | 1.0000 | 1.0000 | 13.0000 | 0.0000 | 0.0000 | 50.0000 |

Generated plots:

- `quality_metrics_vs_threshold.png`
- `final_rows_vs_threshold.png`
