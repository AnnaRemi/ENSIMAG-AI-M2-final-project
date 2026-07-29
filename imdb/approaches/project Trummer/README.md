# Trummer implementations

Exactly two runnable implementations are kept here:

- `baseline/`: vanilla paper-style adaptive semantic block join using one main model.
- `v1/`: structured filtering followed by a cheap-to-expensive two-level batched cascade
  (the formerly named Heterogen V3.2 implementation).

The benchmark flags are `trummer_baseline` and `trummer_v1`.
