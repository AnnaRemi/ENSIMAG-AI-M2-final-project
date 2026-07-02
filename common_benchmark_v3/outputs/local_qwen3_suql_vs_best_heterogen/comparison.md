# Trummer heterogen comparison

| implementation | wall_seconds | llm_calls | cheap_calls | cheap_early_accepts | cheap_early_rejects | expensive_calls | cheap_time_percent | expensive_time_percent | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| suql_baseline | 148.3678 | 25.0000 | 0.0000 | 0.0000 | 0.0000 | 25.0000 | 0.0000 | 100.0000 | 1.0000 | 0.2650 | 0.4161 |
| trummer_heterogen_v2_2_structured_pruned | 8.3260 | 4.0000 | 0.0000 | 0.0000 | 0.0000 | 4.0000 | 0.0000 | 100.0000 | 0.8182 | 0.6923 | 0.7500 |
| trummer_heterogen_v3_pruned_cascade | 14.2799 | 28.0000 | 25.0000 | 24.0000 | 1.0000 | 3.0000 | 68.8716 | 31.1284 | 0.5417 | 1.0000 | 0.7027 |

SUQL applies structured SQL filters before calling answer() on the remaining reviews.
The structured-pruned Trummer variants report their original input size and then apply deterministic year and ID pruning before semantic model calls.
For cascade variants, `llm_calls = cheap_calls + expensive_calls`; non-cascade variants use expensive calls only.

## Routing interpretation

- SUQL issued 25 answer() calls after structured SQL pruning.
- V2_2 issued 4 expensive block-join calls after deterministic pruning.
- V3 issued 25 cheap calls and 3 expensive calls after deterministic pruning.
- V3 early-accepted 24 candidates and early-rejected 1 candidates.
- 0 pruned candidates entered the V3 uncertainty band.
