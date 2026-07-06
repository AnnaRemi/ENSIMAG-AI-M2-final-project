# V2_3 vs V3 vs V3_2 comparison

Run configuration: qwen3:0.6b cheap model, qwen3:1.7b expensive model, 11 repetitions per question.

Best macro F1: V3 (0.589).
Fastest total wall time: V3_2 (40.75s).
Fewest total LLM calls: V3_2 (16 calls).

Per-question observations:
- Q1: best F1 is V3 (0.667); fastest is V3_2 (12.88s).
- Q2: best F1 is V3 (0.545); fastest is V3 (10.48s).
- Q3: best F1 is V3 (0.556); fastest is V3 (11.30s).

Interpretation:
- V3_2 successfully combines SUQL-style structured pruning with batch-wise cascade.
- V3_2 uses far fewer calls than V3, but it is not always faster when it still sends a large expensive fallback batch.
- On Q2, V3_2 is efficient but loses recall, while V3 keeps better quality after pruning.
- On Q3, V3 learns threshold 2.0 and avoids fallback; V3_2 has fewer calls but keeps an expensive fallback batch.
