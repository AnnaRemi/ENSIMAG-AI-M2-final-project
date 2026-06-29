# Common benchmark: three questions with increasing difficulty

## Description

`common_benchmark_3q/` extends the one-question benchmarks into a small
difficulty ladder. It contains three independent 60-row datasets with disjoint
movie IDs and one review per movie. The questions are designed so that
structured pruning alone is insufficient: each dataset includes structured
candidates that are semantic positives and structured candidates that are
semantic negatives.

Use this suite when the question is robustness across query shapes rather than
performance on the single 1998-negative-review task. It compares SUQL,
structured-pruned block join V2_2, batched cascade V2_3, and structured-pruned
cascade V3 across easy, medium, and hard semantic predicates, then reports both
per-question metrics and aggregate macro quality/runtime.

This benchmark contains three independent movie-review retrieval questions with
increasing semantic difficulty. Every question directory contains its own
60-row dataset, annotations, ground truth, and benchmark contract.

| Directory | Structured filters | Semantic task |
| --- | --- | --- |
| `question_1_easy` | `year = 2001` | Overall negative sentiment |
| `question_2_medium` | `genres contains Drama`, `runtime < 100` | Explicit acting/performance praise |
| `question_3_hard` | `genres contains Comedy`, `runtime > 90` | Overall negative despite explicit praise of an aspect |

The easy labels come from the original IMDb 50K sentiment split. Medium and
hard labels are manually curated in `scripts/build_datasets.py`; each semantic
candidate has a saved evidence excerpt and rationale in `annotations.csv`.
No evaluated LLM is used to create ground truth.

Each dataset has:

- 60 unique movie IDs and one review per movie;
- structured candidates containing both semantic positives and negatives;
- distractors that fail at least one structured condition;
- no movie-ID overlap with the other two questions.

## Files

Each question directory contains:

```text
benchmark.json
data/
  annotations.csv
  ground_truth.csv
  imdb_joined.csv
  imdb_reviews.csv
  imdb_structured_joined.csv
```

`manifest.json` summarizes all three questions.

## Rebuild and validate

```bash
cd "/Users/annremizova/Desktop/lab m2"
python3 common_benchmark_3q/scripts/build_datasets.py
python3 -m unittest discover -s common_benchmark_3q/tests -v
```

## Evaluate predictions

Place each method's output at:

```text
predictions/
  question_1_easy/found_rows.csv
  question_2_medium/found_rows.csv
  question_3_hard/found_rows.csv
```

Then run:

```bash
python3 common_benchmark_3q/scripts/evaluate_predictions.py \
  --predictions-dir predictions
```

The evaluator writes per-question precision, recall, and F1 plus macro and
micro aggregates to `predictions/metrics.json`.

## Run the benchmark methods

With a local Ollama server:

```bash
python3 common_benchmark_3q/scripts/run_all.py \
  --python common_benchmark_v3/.venv/bin/python \
  --cheap-model ollama/gemma4:e2b \
  --expensive-model ollama/gemma4:e4b \
  --output-dir outputs/gemma4e
```

This runs SUQL, heterogen_v2_2, heterogen_v2_3, and heterogen_v3 for every
question, then writes `comparison.csv`, `aggregate.csv`, and one
`comparison.png` chart. The retained `outputs/local_llama3.2_qwen2.5_3b/`
artifact is an older partial run with SUQL, heterogen_v2_2, and heterogen_v3.
