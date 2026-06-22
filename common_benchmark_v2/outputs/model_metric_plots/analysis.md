# SUQL baseline vs Trummer: analysis across models

## Scope

1 Ollama model experiments were evaluated on one fixed question with 13 ground-truth movie IDs.

The dataset has 50 unique movie/review rows across 17 years. SUQL applies the year filter before its semantic calls; Trummer receives every row and evaluates year inside the join predicate.

## Per-model results

| Model | SUQL time (s) | Trummer time (s) | Trummer/SUQL | Calls S/T | Rows S/T | SUQL F1 | Trummer F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma2:2b | 0.0 | 0.0 | nanx | 0/0 | 13/13 | 1.000 | 1.000 |

## Aggregate reliability

| Approach | Non-zero runs | Mean precision | Mean recall | Mean F1 | Median F1 | Perfect runs |
| --- | --- | --- | --- | --- | --- | --- |
| SUQL | 1/1 | 1.000 | 1.000 | 1.000 | 1.000 | 1 |
| Trummer | 1/1 | 1.000 | 1.000 | 1.000 | 1.000 | 1 |

## Findings

- SUQL returned non-empty answers for 1/1 models. Trummer returned non-empty answers for 1/1 models.
- SUQL mean F1 was 1.000 and median F1 was 1.000. Trummer mean F1 was 1.000 and median F1 was 1.000.
- Best SUQL F1 was 1.000 with `gemma2:2b`; best Trummer F1 was 1.000 with `gemma2:2b`.
- Mean LLM calls were 0.0 for SUQL and 0.0 for Trummer.
- Trummer was faster in 0/1 paired runs. However, it produced zero rows in 0 of those faster runs.
- Client-process CPU time is not useful for judging inference cost because Ollama executes in a separate process. Engine/wall time and model-server telemetry are the relevant measures.

## Which approach is better?

### Use SUQL when

- answer correctness, stable recall, or predictable behavior matters;
- the chosen model has not been explicitly validated with Trummer's strict index-pair output format;
- false positives are costly;
- the structured year filter reduces the workload from 50 rows to 25 rows;
- results must remain usable across different local models.

### Consider Trummer when

- minimizing the number of API round trips is more important than retrieval accuracy;
- the model is known to follow the exact `x,y` pair format;
- prompts can batch many candidates without exceeding context limits;
- false positives can be verified by a later stage;
- a calibrated fallback reruns malformed or empty outputs using tuple-level classification.

Interpret the v2 results using total latency, token counts, and output quality together. Trummer performs a substantially harder join because the year condition remains inside its LLM predicate.

## Important limitations

- This is one question and one run per model; it does not establish general statistical significance.
- Compare SUQL and Trummer within each model and hardware run; cross-machine timing is not directly comparable.
- Trummer's zero-row outcomes may be formatting/parser failures rather than genuine semantic decisions because raw model responses were not saved.
- LLM call count alone does not measure total token traffic; Trummer prompts contain blocks from a 50x50 candidate cross product.
- The benchmark uses one annotated review per movie, not all reviews for each movie.

## Recommended next experiment

Run at least five repetitions per model and save raw Trummer responses plus prompt/completion tokens for both approaches. Then compare median latency, success rate, total tokens, precision, recall, and F1. Add a Trummer retry/fallback path and report its total cost, not only the initial one-call path.
