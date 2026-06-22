# Stage 2: Cheap-to-Expensive Cascade

Stage 2 implements a two-stage physical operator for SUQL-style `answer(review, question)` filters:

1. A cheap model scores the row as binary `1`/`0`.
2. Confident cheap `Yes` scores are accepted early.
3. Confident cheap `No` scores are rejected early.
4. Unsure scores in the middle band are routed to the expensive full `answer()` model.

This is a local analogue of the Stretto idea of exposing a ladder of physical semantic-operator implementations. Here the ladder is:

```text
cheap score -> early decision when confident -> expensive model when unsure
```

## Files

- `profiler.py`: fits per-question cascade thresholds from labelled `(review, question, label)` examples.
- `cascade_filter.py`: runtime `answer()` implementation used by `src_baseline_stage2`.
- `calibrate.py`: CLI for producing `thresholds.json`.
- `thresholds.json`: default thresholds for the benchmark questions.
- `benchmark_stage2.py`: baseline vs Stage 2 benchmark driver.

## Configuration

Environment variables:

```bash
export SUQL_API_BASE="http://127.0.0.1:11434"
export SUQL_CHEAP_MODEL="ollama/gemma2:2b"
export SUQL_EXPENSIVE_MODEL="ollama/phi4-mini"
export SUQL_MODEL="$SUQL_EXPENSIVE_MODEL"
```

`SUQL_MODEL` is still used by the baseline and parser. Stage 2 uses `SUQL_CHEAP_MODEL` for the scoring step and `SUQL_EXPENSIVE_MODEL` for unsure fallback answers.
Model names can be written either as Ollama names (`gemma2:2b`) or LiteLLM Ollama provider names (`ollama/gemma2:2b`); the runtime normalizes them at each API boundary.

Make sure both models are installed locally, for example:

```bash
ollama pull gemma2:2b
ollama pull phi4-mini
```

`gemma2:2b` is the default cheap model because the saved Stage 2 experiments showed it is a better Ollama scorer for this workload than `llama3.2:1b`. The system can still fall back to `phi4-mini` for uncertain rows, which matters when a query may require sequential calls over hundreds of candidate reviews.

Thresholds are model-specific. If you change `SUQL_CHEAP_MODEL`, regenerate `thresholds.json` with that cheap model before using benchmark quality numbers.

## Try Cheap Models

Before running the full benchmark, smoke-test alternative cheap models against
the same binary scorer used by Stage 2:

```bash
python Stage_2/sweep_cheap_models.py \
  --api-base "$SUQL_API_BASE" \
  --installed-only \
  --models gemma2:2b llama3.2:1b smollm2:360m smollm2:1.7b tinyllama:1.1b
```

To run Stage 2 benchmarks for only the models whose scorer works:

```bash
python Stage_2/sweep_cheap_models.py \
  --api-base "$SUQL_API_BASE" \
  --installed-only \
  --run-benchmarks \
  --expensive-model "$SUQL_EXPENSIVE_MODEL" \
  --models gemma2:2b llama3.2:1b smollm2:360m smollm2:1.7b tinyllama:1.1b
```

The sweep writes logs and CSV results under:

```text
Stage_2/model_sweeps/
Stage_2/benchmarks/<sweep-run-name>_<model>/
```

## Calibrate

Use a labelled CSV with columns:

```text
review,question,label
```

where `label` is `1/0` or `Yes/No`.

```bash
python Stage_2/calibrate.py labelled_examples.csv \
  --output Stage_2/thresholds.json \
  --cheap-model "$SUQL_CHEAP_MODEL" \
  --api-base "$SUQL_API_BASE"
```

The profiler fits one accept threshold and one reject threshold per question using Bayesian lower bounds on decision precision. If no threshold satisfies the requested bound, that side of the cascade is disabled with `inf` or `-inf`.

## Benchmark

```bash
python Stage_2/benchmark_stage2.py \
  --sample-size 100 \
  --seed 11 \
  --api-base "$SUQL_API_BASE" \
  --model "$SUQL_EXPENSIVE_MODEL" \
  --cheap-model "$SUQL_CHEAP_MODEL"
```

Outputs are written under:

```text
Stage_2/benchmarks/<run-name>/
```
