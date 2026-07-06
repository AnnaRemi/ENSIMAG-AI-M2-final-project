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

### `heterogen_v2_2/`

Structured-pruned block semantic join:

- applies deterministic `year=1998` movie filtering before prompting;
- keeps only reviews whose `tconst` matches a selected `movie_id`;
- still uses Trummer-style block prompts for the remaining semantic sentiment
  matching.

See [`heterogen_v2_2/README.md`](heterogen_v2_2/README.md).

### `heterogen_v2_3/`

Batched cheap-to-expensive cascade:

- generates exact `movie_id = tconst` candidates deterministically;
- scores one candidate batch per cheap-model request;
- coalesces uncertain candidates into larger expensive-model batches;
- reports call counts and cheap/expensive model-call time percentages.

See [`heterogen_v2_3/README.md`](heterogen_v2_3/README.md).

### `heterogen_v3/`

Structured-pruned cheap-to-expensive cascade:

- extracts movie-table filters from the question before any model call;
- prunes reviews by exact `movie_id = tconst` keys;
- scores only the pruned exact-ID candidates with the cheap model;
- learns the confidence cutoff from an expensive-model calibration sample;
- sends uncertain candidates to the expensive model, with fallback work capped
  and batched by the runner.

See [`heterogen_v3/README.md`](heterogen_v3/README.md).

## Comparisons

- `../../common_benchmark_v2/` compares heterogeneous v1 with the
  structured-first SUQL baseline.
- `../../common_benchmark_v3/` compares heterogeneous v1, structured-pruned
  v2_2, batched v2_3, pruned-cascade v3, and cascade v2 on the same 50-row
  annotated dataset.

The v1/v2_2 comparison isolates LLM-side predicate evaluation versus
deterministic structured pruning plus block prompts. The v1/v2 comparison
isolates two different semantic-join execution plans: large bounded block
prompts versus exact-ID candidates with model cascading.
The v2/v2_3 comparison isolates row-wise versus batched cascade granularity.
The v2_2/v3 comparison isolates block prompting versus cheap-to-expensive
cascading after the same structured pruning step.
The SUQL/v1 comparison isolates structured-first filtering versus evaluating
the complete predicate inside the semantic join.

## Shared Setup

Install shared dependencies from the project root:

```bash
cd "/Users/annremizova/Desktop/lab m2/project Trummer"
python3 -m pip install -r requirements.txt
```

The per-version runners remain in their implementation directories because they
encode different execution plans. Shared plotting is centralized in
`scripts/plot_metrics.py`.

## Plotting

Generate a bar plot from every discovered Trummer output:

```bash
python3 scripts/plot_metrics.py \
  --metrics elapsed_seconds cheap_calls expensive_calls final_answer_rows \
  --output plots/trummer_metrics.png
```

Filter implementations and choose metrics:

```bash
python3 scripts/plot_metrics.py \
  --impl heterogen_v2 heterogen_v3 heterogen_v3_2 \
  --metrics elapsed_seconds cheap_seconds expensive_seconds \
  --output plots/cascade_time.png
```

Create a tradeoff scatter plot:

```bash
python3 scripts/plot_metrics.py \
  --plot scatter \
  --impl heterogen_v2 heterogen_v3 heterogen_v3_2 \
  --x elapsed_seconds \
  --y final_answer_rows \
  --size expensive_calls \
  --output plots/cascade_tradeoff.png
```

Use `--list-metrics` to print all numeric metrics discovered in the selected
outputs. The loader supports `run_metrics.json`, baseline `*_metrics.json`, and
block-join `use_case3_join_stats.csv` files.

Upstream implementation: <https://github.com/itrummer/llmjoins>

All implementations retain the upstream MIT license where applicable.
