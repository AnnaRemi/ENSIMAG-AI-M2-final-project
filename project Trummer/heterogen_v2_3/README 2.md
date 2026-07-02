# Trummer heterogen_v2_3: batched cascade

This version reduces cheap-model request overhead by classifying several exact
`movie_id = tconst` candidate pairs in one cheap request.

For `N` candidates:

```text
cheap_calls = ceil(N / cheap_batch_size)
expensive_calls = calibration calls + ceil(uncertain_candidates / expensive_batch_size)
```

The defaults are 8 candidates per cheap call and 32 candidates per expensive
call. Because the expensive stage coalesces uncertain candidates across cheap
batches, multi-batch runs have fewer expensive calls than cheap calls even when
every cheap positive requires verification.

The cheap response is a JSON decision for every candidate. Hard decisions are
mapped to `-2/+2`. The cascade learns the confidence threshold from an
expensive-model calibration sample; hardcoded accept/reject cutoffs are not used.
Tune `--cascade-target` and `--calibration-budget` to control the
oracle-agreement target and calibration cost, not the threshold itself.

Every run records:

- `cheap_calls`, `expensive_calls`, calibration calls, and routing counts;
- `cheap_seconds` and `expensive_seconds`;
- `cheap_time_percent` and `expensive_time_percent`, calculated over total
  model-call time.

Run locally:

```bash
cd "/Users/annremizova/Desktop/lab m2/project Trummer/heterogen_v2_3"
python3 -m unittest discover -s tests -v
python3 run_use_case3.py --dry-run --output-dir outputs/local_dry_run
```

Run on Aker:

```bash
cd /home/daisy/remizova/project_Trummer/heterogen_v2_3
CHEAP_BATCH_SIZE=8 EXPENSIVE_BATCH_SIZE=32 bash scripts/run_gpu.sh
```
