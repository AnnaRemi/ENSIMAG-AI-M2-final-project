# SUQL implementations

Exactly two runnable implementations are kept here:

- `baseline/`: vanilla SUQL. Structured SQL runs first; the main model evaluates
  every remaining semantic `answer(review, question)` predicate.
- `v1/`: SUQL Stage 2, implemented as a cheap-to-expensive two-level cascade.

The benchmark flags are `suql_baseline` and `suql_v1`.

`data/` is the source IMDb dataset used to construct benchmark inputs; it is not
an implementation or experiment artifact.
