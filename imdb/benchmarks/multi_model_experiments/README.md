# Multi-model experiments (5q x 10rep)

IMDb and Amazon Fashion each run separately across four cheap/expensive model
pairs, 5 questions x 10 repetitions, four methods per cell. Submitted
2026-07-27 on kraken as jobs 201642-201649; pulled here 2026-07-28.

```
imdb_5q_10rep/<cheap>__<expensive>/     aggregate.csv, comparison.csv, plots/, per_question/
amazon_5q_10rep/<cheap>__<expensive>/   aggregate.csv, comparison.csv, plots/, q_XX/
```

`aggregate.csv` is one averaged row per method; `comparison.csv` is one row per
question x method (IMDb) or per question x method x repetition (Amazon).

## F1 by method

| Pair | dataset | suql_baseline | suql_v1 | trummer_baseline | trummer_v1 |
|---|---|---|---|---|---|
| gemma4:e2b / gemma4:26b | IMDb | 0.520 | 0.448 | 0.425 | 0.411 |
| qwen3.6:27b / qwen3.6:35b | IMDb | 0.512 | 0.585 | 0.382 | 0.131 |
| gemma4:e2b / qwen3.6:35b (cross) | IMDb | 0.512 | 0.477 | 0.382 | 0.104 |
| llama3.1:8b / llama3.1:70b | IMDb | 0.518 | 0.518 | 0.350 | 0.137 |
| gemma4:e2b / gemma4:26b | Amazon | 0.883 | 0.795 | 0.527 | 0.883 |
| qwen3.6:27b / qwen3.6:35b | Amazon | 0.878 | 0.863 | 0.512 | 0.865 |
| gemma4:e2b / qwen3.6:35b (cross) | Amazon | 0.878 | 0.850 | 0.512 | 0.890 |
| llama3.1:8b / llama3.1:70b | Amazon | 0.844 | 0.844 | 0.485 | 0.563 |

## Reading these numbers

**Baselines are a correctness check, not a result.** `suql_baseline` and
`trummer_baseline` only ever call the expensive model, so the qwen-sibling and
cross-family rows share `qwen3.6:35b` and come out byte-identical on both
datasets. They do.

**`trummer_v1` collapses on IMDb for three of four pairs** (0.104-0.137) but not
on Amazon. Gemma is the partial exception at 0.411. Since it spans three model
vendors, this looks like a property of the method on IMDb rather than of any
model. Unexplained as of 2026-07-28.

**`suql_v1` degenerates into `suql_baseline` under llama3.1**, on both datasets:
identical precision/recall/F1 to six decimals, at ~2.5x the calls (98.6 vs 48.6
on Amazon). The cascade scores every candidate cheaply, learns no usable
threshold, falls back to routing everything to the expensive model, and pays
for the discarded cheap pass. A real cascade failure mode, not a bug.

**These runs have no usable error bars.** Every LLM call is temperature 0 and
the calibration draw was deterministic, so repetitions reproduce each other
exactly -- 18 of 20 groups were identical in the Amazon gemma cell. Treat each
cell as n=1. A later fix (`GENERALIZED_CASCADE_SEED`, Amazon pipeline only)
makes repetitions vary; a 3q x 5rep check under it put `trummer_v1` q_03 between
0.750 and 0.968, a spread wider than most between-pair differences in the table
above. Rank cells against each other with that in mind.
