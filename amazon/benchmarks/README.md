# Amazon Fashion benchmarks

Same organisation as [`benchmarks/`](../../benchmarks/README.md) (IMDb): one
runner, four suites, four methods, results kept per suite.

```text
data_amazon/
├── products.csv, reviews.json      # raw source (gitignored)
├── canonical/                      # derived tables (= IMDb's data/)
│   ├── structured.csv, texts.csv, schema.json
├── config.json                     # dataset + layout config
├── questions_10q.json              # reviewed question set
├── questions_heldout_10q.json      # held-out set, disjoint from the above
└── benchmarks/
    ├── build_local_catalog.py      # builds suites + dictionary (no LLM, no GPU)
    ├── run_aker.sh --suite Nq      # cluster submit
    ├── run_local.sh --suite Nq     # local Ollama run
    ├── shared/scripts/             # _aker_worker.sh (the cluster worker)
    ├── semantic_dict/              # mined dictionary context
    ├── scaling/                    # candidate-pool scaling study + scale_n* suites
    └── 1q/ 3q/ 5q/ 10q/
        ├── manifest.json, questions.txt, ground_truth_summary.txt
        ├── per_question/q_XX/benchmark.json
        ├── outputs/<run_name>/     # aggregate.csv, comparison.csv, plots
        ├── logs/
        └── run_aker.sh, run_local.sh   # thin wrappers passing --suite
```

Supported method flags are `suql_baseline`, `suql_v1`, `trummer_baseline` and
`trummer_v1`, exactly as for IMDb.

## Running

```bash
# Build suites and dictionary once (deterministic, local, no GPU)
python3 data_amazon/benchmarks/build_local_catalog.py

# Local
bash data_amazon/benchmarks/5q/run_local.sh --repetitions 10

# Cluster
AKER_HOST=kraken bash data_amazon/benchmarks/5q/run_aker.sh \
  --repetitions 10 --cheap-model gemma4:e2b --expensive-model gemma4:26b
```

## Question sets

`questions_10q.json` is the original reviewed set; the `1q/3q/5q` suites are its
first 1/3/5 questions, so they are nested rather than independent.
`questions_heldout_10q.json` is a genuinely held-out ten — disjoint product
categories and disjoint semantic concepts — mirroring how IMDb's `10q` is
independent of its `1q/3q/5q`.

Building the held-out set without overwriting the existing suites:

```bash
AMAZON_QUESTIONS=data_amazon/questions_heldout_10q.json \
AMAZON_SUITE_ROOT=data_amazon/benchmarks_heldout \
  python3 data_amazon/benchmarks/build_local_catalog.py
```

Ground truth comes from the hand-authored regex patterns in
`build_local_catalog.py` (`SEMANTIC_PATTERNS`), not an LLM judge — every concept
used by a question set needs an entry there. `price` and `category` are not
usable as structured predicates on this dataset: they are 9.5% and 0% populated.

## Layout configuration

`pipeline.py` takes its directory layout from `config.json` rather than assuming
one, so this tree can be rearranged without touching the generic pipeline:

| key | meaning |
|---|---|
| `canonical_dir` | derived `structured.csv` / `texts.csv` / `schema.json` |
| `suites_dir` | parent of the `1q/3q/5q/10q` suite directories |
| `dictionary_dir` | holds `semantic_dictionary.json` |
| `output_dir` | fallback results root; both runners override it per run |

Two environment overrides matter at run time: `GENERALIZED_OUTPUTS_ROOT` keeps
concurrent jobs on the same suite from overwriting each other, and
`GENERALIZED_CASCADE_SEED` controls the per-repetition calibration draw
(`none` restores the deterministic pre-2026-07-27 pick).
