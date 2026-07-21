# Implemented methods

Only four methods are active and benchmark-compatible.

## SUQL baseline (`suql_baseline`)

Location: `project SUQL/baseline/`.

Vanilla structured-first SUQL. Structured SQL predicates are applied before the
main model evaluates each surviving `answer(review, question)` predicate. The
answer parser normalizes leading `Yes`/`No` responses so harmless formatting does
not turn every final row into a rejection.

## SUQL V1 (`suql_v1`)

Location: `project SUQL/v1/`.

The formerly named Stage 2 implementation. A cheap scorer handles confident
decisions and routes uncertain candidates to the expensive model. The directory
is self-contained and includes its cascade filter, scorer, and thresholds.

## Trummer baseline (`trummer_baseline`)

Location: `project Trummer/baseline/`.

Vanilla paper-style adaptive semantic block join. It evaluates movie/review blocks
with the main model and accepts schema-constrained matching movie IDs.

## Trummer V1 (`trummer_v1`)

Location: `project Trummer/v1/`.

The formerly named Heterogen V3.2 implementation: deterministic structured
filtering followed by a cheap-to-expensive batched two-level semantic cascade.

## Benchmark invocation

```bash
bash benchmarks/5q/run_aker.sh \
  --repetitions 3 \
  --methods "suql_v1 trummer_v1" \
  --cheap-model gemma4:e2b \
  --expensive-model gemma4:26b \
  --pull-models
```
