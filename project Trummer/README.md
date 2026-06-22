# Semantic join implementations

This project contains two implementations based on Immanuel Trummer's paper *Implementing Semantic Join Operators Efficiently* and the official `itrummer/llmjoins` repository.

## Approaches

### `baseline/`

Reproduction of the paper's IMDb review experiment:

- official 100-review labeled dataset
- first 50 reviews joined with the next 50
- predicate: both reviews are positive or both are negative
- ground truth derived from the `sentiment` labels
- tuple, block, and adaptive semantic joins
- precision, recall, F1, tokens, prompts, runtime, and estimated cost

See [`baseline/README.md`](baseline/README.md).

### `Heterogen_v1/`

Multi-table heterogeneous retrieval over the local movie and review tables:

- structured movie filtering
- review sampling
- deterministic `movie_id = tconst` prefilter
- Trummer-style block semantic join for the remaining free-text predicate
- local/Ollama execution and Aker OAR workflow
- experiment logs and output CSVs

See [`Heterogen_v1/README.md`](Heterogen_v1/README.md).

## Source

Official implementation: https://github.com/itrummer/llmjoins

Both implementations retain the upstream MIT license.
