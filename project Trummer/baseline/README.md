# Trummer baseline

Paper-style adaptive block join using one main model. The complete question is
placed in the semantic join prompt; no SUQL-style structured pruning or cascade
is used.

Responses use schema-constrained movie IDs and are validated against the
current movie/review blocks. This avoids losing matches because a model used a
different textual output format.

Use the shared benchmark flag `--methods trummer_baseline`.
