# Canonical benchmarks

The four suites share one runner and one implementation registry:

- `10q`: all ten diverse semantic/structured questions.
- `5q`: recommendation/year, fear/genre+runtime, visual/genre, pacing/director, originality/title.
- `3q`: recommendation/year, pacing/director, originality/title.
- `1q`: the recommendation/year smoke test.

Every question contains 100 candidate movies, 40 structured candidates, and 12 ground-truth
movies. Each suite contains `questions.txt`, `ground_truth_movies.txt`, a manifest,
and complete SUQL/Heterogen input CSVs.

Supported implementation flags are `suql_baseline`, `suql_v1`,
`trummer_baseline`, and `trummer_v1`.

Local example (creates `.venv` and runs all four methods against Ollama):

```bash
bash benchmarks/1q/run_local.sh --pull-models
```

Aker example:

```bash
bash benchmarks/5q/run_aker.sh --repetitions 10 --methods "suql_v1 trummer_v1" --pull-models
```

Each run keeps only publication-ready output by default:

- `aggregate.csv`: one averaged row per implementation;
- `comparison.csv`: one row per question and implementation;
- `plots/`: quality, time, call-count, and trade-off plots;
- `per_question/q_XX/plots/`: the same four plots for each question.

Pass `--keep-run-artifacts` directly to `shared/scripts/run_all.py` only when
debugging raw predictions or model responses.

## Dataset annotations

Runtime CSVs live in `data/subdatasets/<suite>/q_XX/`. Every question includes
`annotations.csv`, with one auditable row per candidate movie:

- `structured_match` and `semantic_label` record the two predicate decisions;
- `ground_truth` records their final conjunction;
- `annotation_source` identifies the labeling policy;
- `evidence_excerpt` provides review evidence for the semantic decision;
- `annotation_rationale` explains inclusion or exclusion.

The paired `benchmark.json` records the natural-language query, parsed
structured predicate, semantic predicate, and ground-truth movie IDs.
