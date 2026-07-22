# Semantic-query execution benchmark

This repository compares exactly four semantic-query implementations on the
same IMDb workloads and ground truth:

| Flag | Code | Behavior |
|---|---|---|
| `suql_baseline` | `project SUQL/baseline/` | Vanilla structured-first SUQL; the main model evaluates surviving review predicates. |
| `suql_v1` | `project SUQL/v1/` | Structured-first SUQL with a cheap-to-expensive two-level cascade. |
| `trummer_baseline` | `project Trummer/baseline/` | Vanilla paper-style adaptive semantic block join using the main model. |
| `trummer_v1` | `project Trummer/v1/` | Structured filtering followed by a cheap-to-expensive batched two-level cascade. |

`benchmarks/` is the only experiment tree. Its `10q`, `5q`, `3q`, and `1q`
suites share one implementation registry, runner, evaluator, and plotting code.
Every question has 100 candidate movies, 40 structured candidates, and 12
ground-truth movies.

## Run locally after cloning

Install [Ollama](https://ollama.com/), start it with `ollama serve`, then run
the one-question smoke test. The launcher creates `.venv`, installs the shared
Python dependencies, pulls the two selected models, verifies the local API,
and runs all four implementations:

```bash
git clone git@github.com:AnnaRemi/ENSIMAG-AI-M2-final-project.git
cd ENSIMAG-AI-M2-final-project
bash benchmarks/1q/run_local.sh --pull-models
```

Use `--suite 3q`, `5q`, or `10q` with `benchmarks/run_local.sh` for larger
suites. Models can be changed with `--cheap-model` and `--expensive-model`;
`bash benchmarks/run_local.sh --help` lists every option. Results are written
below `benchmarks/<suite>/outputs/`.

Run any subset on Aker:

```bash
bash benchmarks/run_aker.sh \
  --suite 10q \
  --repetitions 10 \
  --methods "suql_baseline suql_v1 trummer_baseline trummer_v1" \
  --cheap-model gemma4:e2b \
  --expensive-model gemma4:26b \
  --pull-models
```

The runner keeps only the results used for comparison: `aggregate.csv`,
`comparison.csv`, four aggregate plots, and four plots per question. Raw model
responses, repetition directories, logs, and intermediate metrics are removed
after successful evaluation.

The runner supports the lab's complete Gemma 4 family: `gemma4:e2b`,
`gemma4:e4b`, `gemma4:12b`, `gemma4:26b`, and `gemma4:31b`. Baselines use the
expensive model; cascade implementations use both configured models.

## Repository layout

```text
.
├── benchmarks/              # all datasets, experiment runners, tables, and plots
│   ├── {10q,5q,3q,1q}/      # canonical benchmark suites
│   └── shared/scripts/      # four-method runner and evaluator
├── project SUQL/
│   ├── baseline/            # SUQL baseline
│   └── v1/                  # SUQL two-level cascade
├── project Trummer/
│   ├── baseline/            # adaptive semantic block join
│   └── v1/                  # structured two-level cascade
├── docs/                    # method documentation
├── presentations/           # presentation material
└── edbt2027_semantic_query_paper/  # paper source
```

See [`benchmarks/README.md`](benchmarks/README.md) for suite contents and output
details. Each implementation directory contains its own focused README.
