# Trummer heterogen comparison

| implementation | wall_seconds | llm_calls | cheap_calls | cheap_early_accepts | cheap_early_rejects | expensive_calls | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| suql_baseline | 24.8379 | 25 | 0 | 0 | 0 | 25 | 1.0000 | 1.0000 | 1.0000 |
| trummer_heterogen_v1 | 100.5973 | 14 | 0 | 0 | 0 | 14 | 0.3333 | 0.2308 | 0.2727 |
| trummer_heterogen_v2_2_structured_pruned | 28.0452 | 4 | 0 | 0 | 0 | 4 | 0.5000 | 0.2308 | 0.3158 |
| trummer_heterogen_v2_cascade | 127.4784 | 55 | 50 | 0 | 13 | 5 | 0.3636 | 0.6154 | 0.4571 |

The baseline block-join and cascade implementations receive all 50 movies and all 50 reviews.
The structured-pruned variant reports its original input size and then applies deterministic year and ID pruning before the LLM.
For cascade variants, `llm_calls = cheap_calls + expensive_calls`; block-join variants use block calls only.

## Routing interpretation

- V1 issued 14 expensive block-join calls.
- SUQL issued 25 answer() calls after structured SQL pruning.
- V2_2 issued 4 expensive block-join calls after deterministic pruning.
- V2 issued 50 cheap calls and 5 expensive calls.
- V2 early-accepted 0 candidates and early-rejected 13 candidates.
- 37 candidates entered the uncertainty band.
- If expensive calls are zero, the run measures cheap-model routing rather than a meaningful cheap-to-expensive fallback. Threshold calibration is required before treating that configuration as an effective cascade.
