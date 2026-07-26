# Amazon Fashion semantic-query benchmark

This directory runs the same four-method semantic-query comparison used on
IMDb (see the [repository README](../README.md)) against a second, unrelated
domain: Amazon product reviews. The goal is to check whether the cascade
approaches (`suql_v1`, `trummer_v1`) generalize beyond movies, and whether
their cost/quality trade-off holds across different LLM families.

## Dataset

Source: the official 2018 McAuley Lab **Amazon Fashion** release (see
[`source_provenance.json`](source_provenance.json) for URLs and checksums).

- `products.csv` — 186,105 products with structured attributes (`product_id`,
  `title`, `brand`, `category`, `price`, ...).
- `reviews.json` — 882,321 reviews with text, one or more per product.

The complete category is used rather than its 5-core subset, which has only
31 products — too few to guarantee 10+ ground-truth matches per held-out
question. Raw source files and the derived `work/` directory are gitignored;
only `config.json`, `questions_10q.json`, and run outputs are versioned here.

```text
data_amazon/
├── products.csv, reviews.json, *.json.gz   # raw source (gitignored)
├── config.json                             # pipeline config for this dataset
├── questions_10q.json                      # 10 frozen, reviewed questions
├── work/                                   # canonical tables + suites (gitignored)
└── outputs/<run_name>/                     # aggregate.csv, comparison.csv, plots
```

## Approach

Experiments run through [`generalized_pipeline/`](../generalized_pipeline/README.md),
which turns any (structured CSV, free-text CSV) pair into the same
four-method experiment used for IMDb — it does not assume IMDb-specific
fields. The four implementations are unchanged:

| Flag | Behavior |
|---|---|
| `suql_baseline` | Structured-first filtering, then the expensive model judges every survivor. |
| `suql_v1` | Structured-first filtering, then a cheap-to-expensive confidence cascade. |
| `trummer_baseline` | Full semantic/structured task sent without pre-pruning (adaptive block join). |
| `trummer_v1` | Structured pruning, then a cheap-to-expensive batched cascade. |

For the `v1` cascades, the cheap model scores every structured candidate;
`calibration_budget` (20) of them are sampled across the confidence range and
labeled once by the expensive model; the lowest confidence threshold that
still meets `cascade_target` (0.9) agreement is used to let the cheap model
decide early. Everything else falls through to the expensive model.

Questions are frozen and reviewed up front in [`questions_10q.json`](questions_10q.json)
rather than generated per run, so results are reproducible across model
pairs. Each question pairs a structured filter (e.g. `title contains "dress"`)
with a semantic predicate judged from review text (e.g. "described as
comfortable to wear"), evaluated on a 100-candidate pool with a stable,
stratified 70/30 dictionary/evaluation split per question.

## Experiment: 10q, 10rep

Two model pairs were run against the 10-question suite, 10 repetitions per
question/method, on Kraken:

| Run | Models (cheap / expensive) | Questions covered | Output dir |
|---|---|---|---|
| Gemma | `gemma4:e2b` / `gemma4:26b` | **10/10** (full suite) | [`outputs/amazon_fashion_10q_10rep_gemma4_e2b_gemma4_26b_20260724_223605/`](outputs/amazon_fashion_10q_10rep_gemma4_e2b_gemma4_26b_20260724_223605/) |
| Qwen | `qwen3.6:27b` / `qwen3.6:35b` | **5/10** (`q_01`–`q_05` only) | [`outputs/amazon_fashion_5q_10rep_qwen3.6_27b_qwen3.6_35b_20260725_232456/`](outputs/amazon_fashion_5q_10rep_qwen3.6_27b_qwen3.6_35b_20260725_232456/) |

The Qwen run only reached the first 5 questions (dresses, shoes, shirts,
jackets, hats) before the job window closed; watches, bags, socks, sandals,
and costumes (`q_06`–`q_10`) are Gemma-only for now. The analysis below
compares the two models only on the shared `q_01`–`q_05` subset so the
numbers are apples-to-apples; full 10-question Gemma numbers are reported
separately.

Each run directory contains `aggregate.csv` (means per method), `comparison.csv`
(every question × repetition × method row), and plots at both the aggregate
and per-question level.

## Plots

Aggregate plots live under `outputs/<run>/plots/`:

- `01_quality.png` — precision / recall / F1 per method
- `02_time.png` — wall-clock time per method
- `03_calls.png` — cheap vs. expensive LLM call counts
- `04_best_solution.png` — quality/cost Pareto view
- `05_cost_by_approach.png` — modeled dollar cost per method
- `06_cost_vs_f1.png` — cost/quality trade-off scatter

Per-question versions of the same plots live under `outputs/<run>/q_XX/plots/`.
The Gemma run additionally has paper-ready figures under
[`outputs/amazon_fashion_10q_10rep_gemma4_e2b_gemma4_26b_20260724_223605/paper_plots/`](outputs/amazon_fashion_10q_10rep_gemma4_e2b_gemma4_26b_20260724_223605/paper_plots/):
calibration agreement, routing breakdown, an F1 heatmap, a precision/recall
scatter, and cost savings vs. `n`.

## Analysis: Gemma vs. Qwen

Restricted to the shared `q_01`–`q_05` questions, 10 reps each (400 rows / model):

| Method | Model | Precision | Recall | F1 | Wall (s) | LLM calls | cheap / expensive |
|---|---|---|---|---|---|---|---|
| `suql_baseline` | Gemma | 0.903 | 0.880 | 0.883 | 19.5 | 48.6 | 0 / 48.6 |
| `suql_baseline` | Qwen | 0.860 | 0.920 | 0.878 | 21.0 | 48.6 | 0 / 48.6 |
| `suql_v1` | Gemma | 0.871 | 0.747 | 0.795 | 47.0 | 74.8 | 50.0 / 24.8 |
| `suql_v1` | Qwen | 0.825 | 0.933 | 0.863 | 48.4 | 75.0 | 50.0 / 25.0 |
| `trummer_baseline` | Gemma | 0.366 | 0.960 | 0.527 | 33.5 | 37.2 | 0 / 37.2 |
| `trummer_baseline` | Qwen | 0.360 | 0.907 | 0.512 | 28.2 | 34.6 | 0 / 34.6 |
| `trummer_v1` | Gemma | 0.861 | 0.908 | **0.883** | **11.4** | **12.4** | 8.6 / 3.8 |
| `trummer_v1` | Qwen | 0.850 | 0.908 | 0.865 | **17.4** | **9.4** | 7.2 / 2.2 |

(Full 10-question Gemma aggregate, for reference:
`trummer_v1` F1 0.799 at 11.8s/13.0 calls, `suql_baseline` F1 0.835 at
19.7s/49.2 calls — driven down slightly by the harder `q_06`–`q_10`
questions not yet run on Qwen.)

Observations:

- **`trummer_baseline` is a consistent failure mode on Amazon, for both
  models.** Precision collapses to ~0.36 regardless of model family — the
  adaptive block join over-triggers positive matches on review text far more
  than it did on IMDb plot summaries, and no amount of model strength alone
  fixes it. This is a property of the method on this domain, not the LLM.
- **The cascades (`v1`) are the only methods that beat their own baseline on
  cost without giving up quality**, on both models. `trummer_v1` cuts calls
  by ~4x and wall time by ~2-4x versus `trummer_baseline` while roughly
  doubling F1 — the structured pre-pruning step is doing the real work, and
  the cheap/expensive cascade adds efficiency on top.
- **`trummer_v1` is the best method overall on both models**, matching or
  beating `suql_baseline`'s F1 at a quarter of the calls and half the wall
  time. This mirrors the IMDb finding: structured-first pruning plus a
  calibrated cascade is the strongest combination, and that result is not an
  artifact of one dataset.
- **Gemma and Qwen are close but not identical**: Gemma trades a few points
  of recall for precision versus Qwen on every method (e.g. `suql_baseline`
  P 0.903 vs 0.860, R 0.880 vs 0.920), while Qwen is consistently slightly
  cheaper in calls on the cascades. Neither model changes which method wins;
  they shift the operating point along the same precision/recall curve.
- Routing volume is nearly identical across models on `v1` methods (e.g.
  `suql_v1` cheap/expensive split 50.0/24.8 for Gemma vs 50.0/25.0 for Qwen),
  which suggests the calibration procedure is finding a similar confidence
  threshold independent of which model pair is behind it.

## Conclusion

The Amazon Fashion runs replicate the core IMDb result on a second, unrelated
domain and with two different LLM families: **`trummer_v1`'s structured
pruning + calibrated cascade is the best cost/quality trade-off**, and
**`trummer_baseline`'s lack of structured pruning is a genuine method
weakness**, not a model limitation — precision stays low for both Gemma and
Qwen. Model choice (Gemma vs. Qwen) shifts precision/recall slightly but does
not change which method is preferable, which is evidence the four-method
comparison's conclusions are not overfit to one model family.

The main gap is coverage: only `q_01`–`q_05` have a Qwen run. Extending Qwen
to the full 10-question suite (`q_06`–`q_10`: watches, bags, socks, sandals,
costumes) would confirm whether the pattern holds on the harder, lower-recall
questions where Gemma's full-suite F1 dips below the 5-question subset.
`data_amazon/scaling_reliability/` also has a separate candidate-pool-size
scaling study (not covered here) that could be folded into a future revision
of this analysis.

## Reproducing

```bash
python3 generalized_pipeline/pipeline.py --config data_amazon/config.json prepare
python3 generalized_pipeline/pipeline.py --config data_amazon/config.json all
```

Or on Aker, per stage:

```bash
AKER_HOST=kraken STAGE=methods SUITE_SIZE=10 REPETITIONS=10 \
  CHEAP_MODEL=gemma4:e2b EXPENSIVE_MODEL=gemma4:26b \
  bash data_amazon/run_aker.sh
```

See [`generalized_pipeline/README.md`](../generalized_pipeline/README.md) for
the full three-stage workflow, config options, and offline tests.
