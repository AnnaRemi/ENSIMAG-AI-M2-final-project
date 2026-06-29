# Trummer heterogen_v1 vs heterogen_v2 cascade

| implementation | wall_seconds | llm_calls | cheap_calls | cheap_early_accepts | cheap_early_rejects | expensive_calls | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trummer_heterogen_v1 | 113.2794 | 14 | 0 | 0 | 0 | 14 | 0.3333 | 0.1538 | 0.2105 |
| trummer_heterogen_v2_cascade | 114.6514 | 55 | 50 | 0 | 13 | 5 | 0.5000 | 0.1538 | 0.2353 |

Both implementations receive all 50 movies and all 50 reviews under the same semantic predicate.
For v2, `llm_calls = cheap_calls + expensive_calls`; v1 uses block-join calls only.

## Routing interpretation

- V1 issued 14 expensive block-join calls.
- V2 issued 50 cheap calls and 5 expensive calls.
- V2 early-accepted 0 candidates and early-rejected 13 candidates.
- 37 candidates entered the uncertainty band.
- If expensive calls are zero, the run measures cheap-model routing rather than a meaningful cheap-to-expensive fallback. Threshold calibration is required before treating that configuration as an effective cascade.
