# SUQL baseline vs Trummer: analysis across models

## Scope

Seven Ollama model experiments were evaluated on one fixed question with six ground-truth movie IDs. Each model used the same 12 semantic candidates after the year filter.

The `llama3.2` and `phi4-mini` experiments were run locally on the Mac; the other experiments were run on Aker. Therefore, compare SUQL against Trummer within each model, but do not interpret timing differences between models as pure model-speed differences.

## Per-model results

| Model | SUQL time (s) | Trummer time (s) | Trummer/SUQL | Calls S/T | Rows S/T | SUQL F1 | Trummer F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma2:2b | 1022.5 | 603.4 | 0.59x | 12/1 | 5/0 | 0.909 | 0.000 |
| llama3.2 | 27.3 | 32.0 | 1.17x | 12/1 | 8/11 | 0.857 | 0.706 |
| llama3.2:1b | 346.3 | 408.5 | 1.18x | 12/1 | 6/0 | 0.667 | 0.000 |
| mistral:7b | 588.3 | 892.0 | 1.52x | 12/1 | 6/7 | 1.000 | 0.462 |
| phi4-mini | 16.8 | 11.7 | 0.69x | 12/1 | 4/0 | 0.800 | 0.000 |
| qwen2.5:3b | 605.8 | 597.9 | 0.99x | 12/1 | 6/0 | 1.000 | 0.000 |
| qwen2.5:7b | 535.7 | 597.6 | 1.12x | 12/1 | 6/0 | 1.000 | 0.000 |

## Aggregate reliability

| Approach | Non-zero runs | Mean precision | Mean recall | Mean F1 | Median F1 | Perfect runs |
| --- | --- | --- | --- | --- | --- | --- |
| SUQL | 7/7 | 0.917 | 0.881 | 0.890 | 0.909 | 3 |
| Trummer | 2/7 | 0.139 | 0.214 | 0.167 | 0.000 | 0 |

## Findings

- SUQL returned non-empty answers for all 7 models. Trummer returned non-empty answers for only 2: `llama3.2` and `mistral:7b`.
- SUQL mean F1 was 0.890 and median F1 was 0.909. Trummer mean F1 was 0.167 and median F1 was 0.000.
- SUQL achieved perfect precision and recall with `mistral:7b`, `qwen2.5:3b`, and `qwen2.5:7b`.
- Trummer's best result was `llama3.2` with F1 0.706, but it returned 11 rows for six true movies, producing five false positives.
- Trummer with `mistral:7b` returned seven rows but only three true positives, giving F1 0.462.
- Trummer reduced model requests from 12 to 1 for every model. This is a 12x request-count reduction, but not necessarily a token, energy, or monetary-cost reduction because its single request contains all movie and review rows.
- Trummer was faster in 3/7 paired runs. However, it produced zero rows in 3 of those faster runs. It had no paired run that was both faster than SUQL and returned a non-empty answer.
- Client-process CPU time is not useful for judging inference cost because Ollama executes in a separate process. Engine/wall time and model-server telemetry are the relevant measures.

## Which approach is better?

### Use SUQL when

- answer correctness, stable recall, or predictable behavior matters;
- the chosen model has not been explicitly validated with Trummer's strict index-pair output format;
- false positives are costly;
- the structured filter leaves a small or moderate candidate set, as in this benchmark's 12 rows;
- results must remain usable across different local models.

For the current benchmark, SUQL is the clearly better production choice. It dominates Trummer on quality for every tested model and is not consistently slower.

### Consider Trummer when

- minimizing the number of API round trips is more important than retrieval accuracy;
- the model is known to follow the exact `x,y` pair format;
- prompts can batch many candidates without exceeding context limits;
- false positives can be verified by a later stage;
- a calibrated fallback reruns malformed or empty outputs using tuple-level classification.

The current Trummer implementation is experimental rather than robust. Before using it for a larger benchmark, it should persist raw responses, validate output format, retry malformed outputs, and fall back to smaller blocks or per-row classification.

## Important limitations

- This is one question and one run per model; it does not establish general statistical significance.
- Local and Aker hardware timings are mixed. Only within-model SUQL-vs-Trummer timing comparisons are defensible.
- Trummer's zero-row outcomes may be formatting/parser failures rather than genuine semantic decisions because raw model responses were not saved.
- LLM call count alone does not measure total token traffic. SUQL sends 12 smaller prompts; Trummer sends one roughly 3,000-token prompt.
- The benchmark uses one annotated review per movie, not all reviews for each movie.

## Recommended next experiment

Run at least five repetitions per model and save raw Trummer responses plus prompt/completion tokens for both approaches. Then compare median latency, success rate, total tokens, precision, recall, and F1. Add a Trummer retry/fallback path and report its total cost, not only the initial one-call path.
