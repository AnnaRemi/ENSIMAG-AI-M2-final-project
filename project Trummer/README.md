# Trummer Semantic Join Implementations

This directory contains semantic-join implementations based on Immanuel
Trummer's *Implementing Semantic Join Operators Efficiently* and the upstream
`itrummer/llmjoins` project.

## Implementations

### `baseline/`

Reproduction of the paper's labeled IMDb review experiment:

- tuple, block, and adaptive joins;
- a fixed review-pair predicate;
- precision, recall, F1, token, prompt, and runtime metrics.

See [`baseline/README.md`](baseline/README.md).

### `heterogen_v1/`

Bounded block semantic join over heterogeneous movie and review tables:

- movie and review rows are partitioned into prompt-sized blocks;
- the LLM evaluates identity, structured constraints, and review semantics;
- responses use schema-constrained `matching_movie_ids`;
- raw responses and parsed-pair counts are retained for diagnosis.

See [`heterogen_v1/README.md`](heterogen_v1/README.md).

### `heterogen_v2/`

Exact-ID candidate generation followed by a cheap-to-expensive cascade:

- deterministic `movie_id = tconst` candidate generation;
- cheap binary scoring for every candidate;
- early decisions at configured thresholds;
- expensive-model verification for uncertain candidates.

See [`heterogen_v2/README.md`](heterogen_v2/README.md).

## Comparisons

- `../../common_benchmark_v2/` compares heterogeneous v1 with the
  structured-first SUQL baseline.
- `../../common_benchmark_v3/` compares heterogeneous v1 with heterogeneous v2
  on the same 50-row annotated dataset.

The v1/v2 comparison isolates two different semantic-join execution plans:
large bounded block prompts versus exact-ID candidates with model cascading.
The SUQL/v1 comparison isolates structured-first filtering versus evaluating
the complete predicate inside the semantic join.

Upstream implementation: <https://github.com/itrummer/llmjoins>

All implementations retain the upstream MIT license where applicable.
