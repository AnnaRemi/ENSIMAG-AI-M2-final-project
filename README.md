# Heterogeneous Question Answering over Structured and Unstructured Data using Large Language Models

Master 2 research workspace for comparing LLM-backed query execution over
structured IMDb metadata and unstructured movie reviews.

The retained work has two focuses:

1. SUQL structured-first execution: baseline, Stage 1 calibrated early exits,
   and Stage 2 cheap-to-expensive routing.
2. Semantic joins: Trummer heterogeneous v1 versus heterogeneous v2, and
   Trummer heterogeneous v1 versus the SUQL baseline on shared annotated data.

## Retained Systems

| System | Location | Execution strategy |
| --- | --- | --- |
| SUQL baseline | `project SUQL/src_baseline/` | Apply structured predicates first, then evaluate the semantic `answer()` predicate on surviving reviews. |
| SUQL Stage 1 | `project SUQL/src_baseline_stage1/`, `project SUQL/Stage_1/` | Score every candidate, accept or reject confident rows early, and use full generation for the ambiguous band. |
| SUQL Stage 2 | `project SUQL/src_baseline_stage2/`, `project SUQL/Stage_2/` | Use a cheap binary scorer and route uncertain candidates to an expensive full-answer model. |
| Trummer heterogeneous v1 | `project Trummer/heterogen_v1/` | Evaluate movie/review blocks with bounded semantic-join prompts and schema-constrained outputs. |
| Trummer heterogeneous v2 | `project Trummer/heterogen_v2/` | Generate exact-ID candidates, score them cheaply, and verify uncertain candidates with an expensive model. |

Stage 1 and Stage 2 are experimental physical operators inspired by calibrated
cascades and Stretto-style operator selection. They are not complete
implementations of the Stretto execution engine.

## Main Comparisons

### Trummer heterogeneous v1 versus v2

`common_benchmark_v3/` compares both Trummer implementations on the same
50-movie, 50-review dataset. The shared predicate requires:

1. movie year is 1998;
2. `movie_id = tconst`;
3. the review expresses a negative or strongly critical opinion.

V1 sends bounded cross-product blocks to one model. V2 first creates exact-ID
candidates, then applies a cheap-to-expensive cascade.

For the retained `llama3.2 -> qwen2.5:3b` experiment:

| Metric | Heterogeneous v1 | Heterogeneous v2 |
| --- | ---: | ---: |
| Wall time | 96.66 s | 110.69 s |
| Total LLM calls | 14 | 55 |
| True positives | 3 | 8 |
| False positives | 6 | 14 |
| Recall | 0.231 | 0.615 |
| F1 | 0.273 | 0.457 |

The cascade improved recall and F1, but it was slower and issued more total
calls. The result demonstrates a quality/cost tradeoff, not a universal win for
v2.

![Trummer heterogeneous v1 versus v2](common_benchmark_v3/outputs/cheap_llama3.2__expensive_qwen2.5_3b/comparison.png)

Source metrics:
[`comparison.csv`](common_benchmark_v3/outputs/cheap_llama3.2__expensive_qwen2.5_3b/comparison.csv).

### Trummer heterogeneous v1 versus SUQL baseline

`common_benchmark_v2/` compares the fixed-output heterogeneous v1 operator with
the SUQL baseline on the same 50 rows and 13 ground-truth movie IDs.

For the retained local `qwen2.5:3b` experiment:

| Metric | SUQL baseline | Heterogeneous v1 |
| --- | ---: | ---: |
| Engine time | 47.83 s | 203.59 s |
| LLM calls | 25 | 14 |
| True positives | 13 | 2 |
| False positives | 0 | 5 |
| False negatives | 0 | 11 |
| Precision | 1.000 | 0.286 |
| Recall | 1.000 | 0.154 |
| F1 | 1.000 | 0.200 |

All 14 v1 requests returned valid structured JSON and none overflowed.
Therefore, this run's low quality is attributable to semantic block decisions,
not to the former positional-output parsing problem. SUQL performed more calls,
but each call evaluated one review after the structured year filter; v1 used
fewer but much larger block prompts over all 50 movies and reviews.

![SUQL versus Trummer quality and workload](common_benchmark_v2/outputs/qwen2.5_3b/workload_quality_comparison.png)

Source metrics:
[`comparison.csv`](common_benchmark_v2/outputs/qwen2.5_3b/comparison.csv) and
[`join_stats.csv`](common_benchmark_v2/outputs/qwen2.5_3b/trummer_heterogen_v1/join_stats.csv).

`common_benchmark/` retains the earlier 16-row multi-model comparison. It is
useful for model-sensitivity analysis, while `common_benchmark_v2/` is the
preferred mixed-year comparison because the Trummer predicate must enforce the
year condition itself.

## SUQL Baseline, Stage 1, and Stage 2

The SUQL subproject now contains only the three retained execution strategies
and their supporting experiments.

### Baseline

The baseline applies ordinary SQL-compatible predicates before semantic review
evaluation. This reduces the number of rows sent to the LLM and provides the
reference behavior for Stage 1 and Stage 2.

### Stage 1

Stage 1 adds a calibrated scorer before full generation:

```text
score >= accept threshold  -> Yes
score <= reject threshold  -> No
otherwise                  -> full answer() call
```

Its benefit depends on scorer cost and the fraction of candidates decided
early. Ambiguous candidates pay for both scoring and full generation.

![Baseline versus Stage 1 scaling](project%20SUQL/Stage_1/benchmarks/baseline_vs_stage1_data_samples_20260616_110847/metrics_vs_sample_size.svg)

### Stage 2

Stage 2 separates the cheap scorer from the expensive answer model. It can
reject, accept, skip cheap scoring when observed yield is too low, or fall back
to the expensive model. Results must therefore be interpreted using cheap
calls, early decisions, skips, and expensive calls—not wall time alone.

Retained Stage 2 experiments are under `project SUQL/Stage_2/benchmarks/`.

## Repository Structure

```text
.
├── project SUQL/
│   ├── src_baseline/             # structured-first reference engine
│   ├── src_baseline_stage1/      # calibrated early-exit engine
│   ├── src_baseline_stage2/      # cheap-to-expensive cascade engine
│   ├── Stage_1/                  # calibration and baseline-vs-Stage-1 experiments
│   ├── Stage_2/                  # cascade and baseline-vs-Stage-2 experiments
│   ├── data/                     # local IMDb data preparation
│   ├── data_samples/             # deterministic benchmark samples
│   └── scripts/                  # retained Stage 1/Stage 2 analysis and Aker runners
├── project Trummer/
│   ├── baseline/                 # paper-style semantic join reproduction
│   ├── heterogen_v1/             # bounded block semantic join
│   └── heterogen_v2/             # exact-ID cascade semantic join
├── common_benchmark/             # legacy SUQL baseline vs v1 model sweep
├── common_benchmark_v2/          # mixed-year SUQL baseline vs fixed-output v1
├── common_benchmark_v3/          # v1 vs v2 cascade
├── presentations/                # final project and paper-review slides
└── papers/                       # local reading material, not tracked
```

## Setup

Requirements:

- Python 3.11 or newer;
- Ollama;
- enough memory for the selected local model;
- IMDb-derived data files for full SUQL experiments.

```bash
cd "/path/to/lab m2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r "project SUQL/requirements.txt"
pip install -r common_benchmark_v3/requirements.txt
```

Install representative models:

```bash
ollama pull gemma2:2b
ollama pull phi4-mini
ollama pull qwen2.5:3b
```

Configure the local endpoint:

```bash
export SUQL_API_BASE="http://127.0.0.1:11434"
export SUQL_MODEL="ollama/phi4-mini"
export SUQL_CHEAP_MODEL="ollama/gemma2:2b"
export SUQL_EXPENSIVE_MODEL="ollama/phi4-mini"
```

## Running SUQL Experiments

Run the baseline:

```bash
cd "project SUQL/src_baseline"
python main.py \
  "Which horror movies under 110 minutes have reviews mentioning suspense or tension?"
```

Run baseline versus Stage 1:

```bash
cd "project SUQL"
python Stage_1/benchmark_stage1.py \
  --sample-size 100 \
  --model ollama/phi4-mini \
  --api-base "$SUQL_API_BASE"
```

Run baseline versus Stage 2:

```bash
cd "project SUQL"
python Stage_2/benchmark_stage2.py \
  --sample-size 100 \
  --seed 11 \
  --api-base "$SUQL_API_BASE" \
  --model "$SUQL_EXPENSIVE_MODEL" \
  --cheap-model "$SUQL_CHEAP_MODEL"
```

Stage 1 and Stage 2 thresholds are model-specific. Recalibrate them after
changing the scorer model, prompt, or data distribution.

## Running Shared Comparisons

Run SUQL baseline versus Trummer heterogeneous v1:

```bash
cd "/path/to/lab m2"
"project SUQL/.venv/bin/python" common_benchmark_v2/scripts/run_all.py \
  --api-base "$SUQL_API_BASE" \
  --model ollama/qwen2.5:3b \
  --skip-build-dataset
```

Run Trummer heterogeneous v1 versus v2:

```bash
python3 -m unittest discover -s common_benchmark_v3/tests -v

python3 common_benchmark_v3/scripts/run_all.py \
  --api-base "$SUQL_API_BASE" \
  --cheap-model ollama/llama3.2 \
  --expensive-model ollama/qwen2.5:3b
```

Each common benchmark contains local execution, Aker synchronization, OAR
submission, progress monitoring, and result-retrieval scripts. See its README
for the exact workflow and output contract.

## Reading Results

- Do not compare dry-run metrics with real LLM runs.
- Use metrics sidecars when scorer calls are separate from engine logs.
- Count cheap scoring, expensive fallback, and downstream work.
- Compare retrieval quality together with prompts and wall time.
- Treat saved results as specific to their dataset, prompt, model, threshold,
  and hardware configuration.

## References

- Liu et al., *SUQL: Conversational Search over Structured and Unstructured
  Data with Large Language Models*.
- Trummer et al., *Implementing Semantic Join Operators Efficiently*.
- Stretto work on physical operators for LLM-augmented data systems.
