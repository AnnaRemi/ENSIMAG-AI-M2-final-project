# Common benchmark v3: Trummer heterogen_v1 vs heterogen_v2

This benchmark compares only:

- `project Trummer/heterogen_v1`: bounded Trummer block join, with no cascade;
- `project Trummer/heterogen_v2`: exact-ID candidate generation followed by a
  cheap-to-expensive cascade.

It reuses the fixed 50-row dataset and 13-ID ground truth from
`common_benchmark_v2`.

## Fair comparison contract

Both implementations receive all 50 movie rows and all 50 review rows. The
shared predicate requires:

1. movie year is 1998;
2. `movie_id = tconst`;
3. the review is negative, critical, or strongly unfavorable.

V1 evaluates the predicate through bounded block-join prompts. V2 first creates
the 50 exact-ID candidates, scores each candidate with the cheap model, and
sends only uncertain candidates to the expensive model.

V1 returns schema-constrained `matching_movie_ids` copied from each movie block.
This avoids positional pair-index failures while preserving the block-join
execution strategy. Its `join_stats.csv` includes raw responses and parsed-pair
counts.

## Local validation

```bash
cd "/Users/annremizova/Desktop/lab m2"
python3 -m unittest discover -s common_benchmark_v3/tests -v
python3 common_benchmark_v3/scripts/run_all.py \
  --cheap-model ollama/llama3.2 \
  --expensive-model ollama/qwen2.5:3b \
  --dry-run
```

Run the real local benchmark:

```bash
python3 common_benchmark_v3/scripts/run_all.py \
  --api-base http://127.0.0.1:11434 \
  --cheap-model ollama/llama3.2 \
  --expensive-model ollama/qwen2.5:3b
```

Outputs are written under:

```text
common_benchmark_v3/outputs/cheap_<cheap>__expensive_<expensive>/
```

The directory contains per-implementation evidence and metrics plus
`comparison.csv`, `comparison.md`, `movie_id_outcomes.csv`, and
`comparison.png`.

## Aker GPU

Local Mac:

```bash
cd "/Users/annremizova/Desktop/lab m2"
bash common_benchmark_v3/scripts/sync_common_benchmark_to_aker.sh
```

Aker login node:

```bash
cd /home/daisy/remizova/common_benchmark_v3_workspace
CHEAP_MODEL=gemma2:2b \
EXPENSIVE_MODELS="qwen2.5:3b" \
PULL_MODELS=1 \
WALLTIME=08:00:00 \
bash common_benchmark_v3/scripts/submit_aker_common_benchmark.sh
```

Progress on Aker:

```bash
oarstat -u "$USER"
oarstat -f -j <jobid>
tail -F common_benchmark_v3/logs/oar_<jobid>.out
tail -F common_benchmark_v3/logs/*_<jobid>_*.console.log
```

Local Mac after completion:

```bash
bash common_benchmark_v3/scripts/pull_common_benchmark_from_aker.sh
```

Thresholds default to `reject <= -1.5` and `accept >= 3.0`. With hard `±2`
scores, v2 cheaply rejects negative decisions and sends positive decisions to
the expensive model for verification. They remain operational defaults, not
calibrated benchmark conclusions.
