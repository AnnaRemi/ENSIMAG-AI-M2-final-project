# Heterogeneous Question Answering over Structured and Unstructured Data using Large Language Models

Master 2 research workspace for comparing LLM-backed query execution over
structured IMDb metadata and unstructured movie reviews.

The retained work has three focuses:

1. SUQL structured-first execution: baseline, Stage 1 calibrated early exits,
   and Stage 2 cheap-to-expensive routing.
2. Semantic joins: Trummer heterogeneous block joins, structured pruning,
   row-wise cascades, and batched cascades on shared annotated data.
3. Benchmark suites for one-question comparisons, threshold sweeps, and
   three-question difficulty scaling.

## Retained Systems

| System | Location | Execution strategy |
| --- | --- | --- |
| SUQL baseline | `project SUQL/src_baseline/` | Apply structured predicates first, then evaluate the semantic `answer()` predicate on surviving reviews. |
| SUQL Stage 1 | `project SUQL/src_baseline_stage1/`, `project SUQL/Stage_1/` | Score every candidate, accept or reject confident rows early, and use full generation for the ambiguous band. |
| SUQL Stage 2 | `project SUQL/src_baseline_stage2/`, `project SUQL/Stage_2/` | Use a cheap binary scorer and route uncertain candidates to an expensive full-answer model. |
| Trummer heterogeneous v1 | `project Trummer/heterogen_v1/` | Evaluate movie/review blocks with bounded semantic-join prompts and schema-constrained outputs. |
| Trummer heterogeneous v2 | `project Trummer/heterogen_v2/` | Generate exact-ID candidates, score them cheaply, and verify uncertain candidates with an expensive model. |
| Trummer heterogeneous v2_2 | `project Trummer/heterogen_v2_2/` | Prune year and exact movie/review IDs deterministically, then run Trummer block prompts for sentiment matching. |
| Trummer heterogeneous v2_3 | `project Trummer/heterogen_v2_3/` | Score exact-ID candidates in cheap batches and coalesce uncertain candidates into larger expensive batches. |
| Trummer heterogeneous v3 | `project Trummer/heterogen_v3/` | Combine structured pruning with the cheap-to-expensive cascade. |

Stage 1 and Stage 2 are experimental physical operators inspired by calibrated
cascades and Stretto-style operator selection. They are not complete
implementations of the Stretto execution engine.

## Implementation Variants

The Trummer variants represent different physical plans for the same
movie-review semantic join:

| Variant | Structured pruning | Candidate unit | Cheap model use | Expensive model use |
| --- | --- | --- | --- | --- |
| V1 block join | None | Movie block x review block | None | Evaluates identity, year, and sentiment inside each block prompt. |
| V2 row-wise cascade | Exact `movie_id = tconst` join only | One movie-review pair | Scores every exact-ID pair independently. | Verifies uncertain pairs in fallback batches. |
| V2_2 structured-pruned block join | Question-derived movie filters plus exact review IDs | Pruned movie block x review block | None | Evaluates only the remaining semantic review predicate. |
| V2_3 batch-wise cascade | Exact `movie_id = tconst` join only | Batch of exact-ID pairs | Scores multiple pairs per cheap request. | Coalesces uncertain candidates into larger expensive batches. |
| V3 pruned cascade | Question-derived movie filters plus exact review IDs | Pruned exact-ID pair | Scores only candidates that survive structured pruning. | Verifies uncertain pruned candidates, capped by `--max-expensive-calls`. |

V2_2 and V3 use the structured predicate extractor from the newer heterogeneous
implementations. It maps question constraints onto the movie schema
(`movie_id`, `title`, `director`, `year`, `runtime`, and `genres`) and leaves
review sentiment or opinion matching to the LLM. V2_3 changes request
granularity rather than the predicate semantics: it keeps the same exact-ID
candidate set as V2, but reduces request overhead by classifying and verifying
batches.

## Main Comparisons

### SUQL baseline versus best heterogeneous variants

`common_benchmark_v3/` compares the SUQL baseline with the two strongest
heterogeneous plans from the Trummer family on the same 50-movie, 50-review
dataset. The shared task requires:

1. movie year is 1998;
2. `movie_id = tconst`;
3. the review expresses a negative or strongly critical opinion.

The local comparison below uses `qwen3:1.7b` for SUQL and every expensive-model
stage. The structured parser and cheap cascade stage use `qwen3:0.6b`. Each
experiment is run 9 times and numeric metrics are averaged.

| Version | Wall time | Total LLM calls | Cheap calls | Expensive calls | Final rows | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SUQL baseline | 148.37 s | 25 | 0 | 25 | 3.44 | 1.000 | 0.265 | 0.416 |
| Structured pruning block join | 8.33 s | 4 | 0 | 4 | 11.00 | 0.818 | 0.692 | 0.750 |
| Structured pruning cascade | 14.28 s | 28 | 25 | 3 | 24.00 | 0.542 | 1.000 | 0.703 |

SUQL issued 25 expensive `answer()` calls after SQL pruning. With the qwen3
answer parser fixed, it is very precise in this run but conservative: it returns
about 3.4 rows on average and misses most ground-truth movies.

The structured-pruning block join is the most efficient plan here. It uses only
4 expensive calls after deterministic year and ID pruning, runs much faster than
SUQL, and has the highest F1. The structured-pruning cascade reaches full recall
by routing candidates through 25 cheap calls and only 3 expensive calls, but it
accepts more false positives, so its precision and F1 are lower than the block
join variant.

![Precision, recall, and F1 for SUQL versus best heterogeneous variants](common_benchmark_v3/outputs/local_qwen3_suql_vs_best_heterogen/metrics_precision_recall_f1.png)

![Wall time for SUQL versus best heterogeneous variants](common_benchmark_v3/outputs/local_qwen3_suql_vs_best_heterogen/time_bar_plot.png)

![LLM calls for SUQL versus best heterogeneous variants](common_benchmark_v3/outputs/local_qwen3_suql_vs_best_heterogen/calls_bar_plot.png)

Source metrics:
[`comparison.csv`](common_benchmark_v3/outputs/local_qwen3_suql_vs_best_heterogen/comparison.csv).

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

### Threshold and three-question suites

`common_benchmark_thresholds/` sweeps the cascade confidence threshold for V2
and V2_3. The newest saved run uses `qwen3:0.6b -> qwen3:1.7b` over thresholds
from `0` to `3`, averaged across 9 repetitions.

![Cascade quality versus threshold](common_benchmark_thresholds/outputs/threshold_sweep__cheap_qwen3_0.6b__expensive_qwen3_1.7b/quality_metrics_vs_threshold.png)

`common_benchmark_3q/` adds three 60-row movie-review questions with increasing
semantic difficulty. The retained aggregate plot compares SUQL, structured
pruned block join V2_2, and pruned cascade V3.

![Three-question benchmark aggregate](common_benchmark_3q/outputs/local_llama3.2_qwen2.5_3b/comparison.png)

## Experiment Suites

| Suite | Dataset and comparison | Use it for | Main runner | Primary outputs |
| --- | --- | --- | --- | --- |
| `common_benchmark/` | 16 unique movies for the 1998 negative-review task; SUQL baseline versus Trummer V1 block join. | Compact model-sensitivity runs across one non-cascading model at a time. | `python3 common_benchmark/scripts/run_all.py` | `comparison.csv`, `comparison.md`, `movie_id_outcomes.csv`, time and workload/quality plots, cross-model plots. |
| `common_benchmark_v2/` | 50 mixed-year movies for the same task; SUQL baseline versus Trummer V1 block join. | Preferred SUQL-vs-V1 test because Trummer must evaluate year, identity, and sentiment over all rows. | `python3 common_benchmark_v2/scripts/run_all.py` | `comparison.csv`, `comparison.md`, `movie_id_outcomes.csv`, time and workload/quality plots. |
| `common_benchmark_v3/` | Reuses the v2 50-row dataset; SUQL plus V1, V2, V2_2, V2_3, and V3. | Comparing physical semantic-join plans under one fixed question. | `python3 common_benchmark_v3/scripts/run_all.py` | `comparison.csv`, `comparison.md`, `movie_id_outcomes.csv`, focused quality/time/call plots. |
| `common_benchmark_v3/` all-Heterogen | Same v2 50-row dataset, but only V1, V2, V2_2, V2_3, and V3 with repetition averaging and Aker helpers. | Isolating Trummer heterogen variants without SUQL in the result set. | `python3 common_benchmark_v3/scripts/run_all_heterogen.py` | `all_metrics.csv`, `summary.md`, `experiment_config.json`, per-implementation `run_metrics_repetitions.csv`, focused plots. |
| `common_benchmark_thresholds/` | Same v3 one-question dataset; manual confidence-threshold sweep for V2 and V2_3. | Understanding how threshold choice changes quality, final rows, early decisions, and fallback load. | `python3 common_benchmark_thresholds/scripts/run_threshold_sweep.py` | `threshold_metrics.csv`, `summary.md`, quality-vs-threshold and final-rows plots. |
| `common_benchmark_3q/` | Three disjoint 60-row datasets with easy, medium, and hard semantic predicates; SUQL, V2_2, V2_3, and V3. | Testing robustness beyond the single 1998-negative-review task. | `python3 common_benchmark_3q/scripts/run_all.py` | Per-question run folders, `comparison.csv`, `aggregate.csv`, `comparison.png`. |

The benchmark runners preserve per-run artifacts rather than only aggregate
tables. Cascade runs keep `cascade_decisions.csv`; block-join runs keep
`joined_evidence.csv` and `join_stats.csv`; repeated runs keep
`run_metrics_repetitions.csv` next to the averaged `run_metrics.json`.

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
│   ├── heterogen_v2/             # pair-level exact-ID cascade semantic join
│   ├── heterogen_v2_2/           # structured-pruned bounded block semantic join
│   ├── heterogen_v2_3/           # batched exact-ID cascade semantic join
│   └── heterogen_v3/             # structured-pruned cascade semantic join
├── common_benchmark/             # legacy SUQL baseline vs v1 model sweep
├── common_benchmark_v2/          # mixed-year SUQL baseline vs fixed-output v1
├── common_benchmark_v3/          # one-question heterogen implementation comparison
├── common_benchmark_thresholds/   # cascade-threshold sweep
├── common_benchmark_3q/           # three-question difficulty benchmark
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
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
ollama pull qwen2.5:3b
```

Configure the local endpoint:

```bash
export SUQL_API_BASE="http://127.0.0.1:11434"
export SUQL_MODEL="ollama/qwen3:1.7b"
export SUQL_CHEAP_MODEL="ollama/qwen3:0.6b"
export SUQL_EXPENSIVE_MODEL="ollama/qwen3:1.7b"
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

Run the Trummer heterogeneous variants:

```bash
python3 -m unittest discover -s common_benchmark_v3/tests -v

python3 common_benchmark_v3/scripts/run_all_heterogen.py \
  --api-base "$SUQL_API_BASE" \
  --cheap-model "$SUQL_CHEAP_MODEL" \
  --expensive-model "$SUQL_EXPENSIVE_MODEL" \
  --repetitions 9
```

Run SUQL plus the heterogen variants on the same one-question benchmark:

```bash
python3 common_benchmark_v3/scripts/run_all.py \
  --api-base "$SUQL_API_BASE" \
  --cheap-model "$SUQL_CHEAP_MODEL" \
  --expensive-model "$SUQL_EXPENSIVE_MODEL" \
  --repetitions 9
```

Run the threshold sweep for V2 and V2_3:

```bash
python3 common_benchmark_thresholds/scripts/run_threshold_sweep.py \
  --api-base "$SUQL_API_BASE" \
  --cheap-model "$SUQL_CHEAP_MODEL" \
  --expensive-model "$SUQL_EXPENSIVE_MODEL" \
  --thresholds 0,0.5,1,1.5,2,2.5,3 \
  --repetitions 9
```

Run the three-question suite:

```bash
python3 common_benchmark_3q/scripts/run_all.py \
  --api-base "$SUQL_API_BASE" \
  --cheap-model "$SUQL_CHEAP_MODEL" \
  --expensive-model "$SUQL_EXPENSIVE_MODEL" \
  --output-dir outputs/qwen3_current
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

The project uses the following literature as design and comparison context.
Local PDF copies may exist under `papers/`, but that directory is intentionally
not tracked in Git.

| Work | Used for | Link |
| --- | --- | --- |
| Shicheng Liu, Jialiang Xu, Wesley Tjangnaka, Sina Semnani, Chen Yu, and Monica Lam. *SUQL: Conversational Search over Structured and Unstructured Data with Large Language Models*. Findings of NAACL 2024. | Structured-first query execution, `answer()`/`summary()` semantics, and the SUQL baseline. | <https://aclanthology.org/2024.findings-naacl.283/> |
| Immanuel Trummer. *Implementing Semantic Join Operators Efficiently*. arXiv:2510.08489. | Tuple, block, and adaptive semantic-join execution plans; basis for Trummer baseline and heterogeneous block joins. | <https://arxiv.org/abs/2510.08489> |
| Gabriele Sanmartino, Matthias Urban, Paolo Papotti, and Carsten Binnig. *The Stretto Execution Engine for LLM-Augmented Data Systems*. arXiv:2602.04430. | Physical-operator selection, runtime-quality tradeoffs, and error-budget/cascade framing for Stage 1, Stage 2, and heterogen cascades. | <https://arxiv.org/abs/2602.04430> |
| Matthias Urban and Carsten Binnig. *ELEET: Efficient Learned Query Execution over Text and Tables*. arXiv:2410.22522. | Learned query execution over mixed text/table data and comparison point for non-LLM or smaller-model execution. | <https://arxiv.org/abs/2410.22522> |
| Xuanhe Zhou, Junxuan He, Wei Zhou, Haodong Chen, Zirui Tang, Haoyu Zhao, Xin Tong, Guoliang Li, Youmin Chen, Jun Zhou, Zhaojun Sun, Binyuan Hui, Shuo Wang, Conghui He, Zhiyuan Liu, Jingren Zhou, and Fan Wu. *A Survey of LLM x DATA*. arXiv:2505.18458. | Broader LLM4Data/Data4LLM taxonomy used in presentation and project positioning. | <https://arxiv.org/abs/2505.18458> |

Additional source material:

- SUQL implementation reference: <https://github.com/stanford-oval/suql>.
- Trummer semantic-join implementation reference and IMDb review sample:
  <https://github.com/itrummer/llmjoins>.
- IMDb 50K review dataset cited by the Trummer baseline data provenance:
  <https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert>.
