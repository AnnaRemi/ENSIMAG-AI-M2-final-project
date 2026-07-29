# SUQL baseline

Vanilla SUQL-style execution for the movie benchmark:

1. execute structured SQL predicates first;
2. evaluate the remaining review predicate with the main model;
3. return the matching movie rows.

The parser accepts harmless answer formatting such as `Answer: Yes` in addition
to `Yes`. This prevents valid positive model answers from being discarded while
leaving the baseline architecture unchanged.

Use the shared benchmark flag `--methods suql_baseline`.
