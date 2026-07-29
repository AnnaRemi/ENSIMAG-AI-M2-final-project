# Trummer baseline

Paper-style adaptive block join using one main model. Each adaptive block pair
is evaluated in one semantic-join prompt containing multiple tuples. The
complete question and its mined semantic guideline are placed in that prompt;
no SUQL-style structured pruning or cascade is used.

Responses use a simple line-oriented ID/decision protocol with tolerant parsing,
so execution does not depend on provider-specific structured-output support.

Use the shared benchmark flag `--methods trummer_baseline`.
