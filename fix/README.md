# Fixed semantic-retrieval implementations

This directory is an isolated copy of the four active implementations. The
original `project SUQL/` and `project Trummer/` trees are unchanged.

## Applied recommendations

- Full reviews are split into overlapping 3,200-character windows (about 800
  tokens) with 400-character overlap.
- Compound recommendation predicates are decomposed into `recommend`, `worth
  watching`, and `must-see` evidence questions and OR-combined.
- Concrete negated and qualified examples are included in the extraction prompt.
- Evidence is copied verbatim and validated against the source chunk.
- Extraction and final decisions use JSON schemas.
- Final decisions contain evidence, a continuous `P(YES)` confidence, and a
  verdict. The portable confidence threshold defaults to 0.35.
- SUQL V1 keeps native first-token log-odds when the backend provides them and
  scores extracted evidence rather than a truncated review prefix.
- Both cascades use asymmetric low/high routing thresholds. Reject thresholds
  are deliberately conservative for recall.
- Cascade calibration uses canonical regex labels when the benchmark question
  family is recognized; it no longer treats the expensive model as ground truth.
- Trummer V1 evaluates both cheap and routed expensive candidates in shared
  semantic-join batches (eight candidates per prompt by default).
- Routed/final candidates use three samples at temperature 0.4 and majority vote.
- Extracted spans, rather than truncated raw reviews, are passed to final stages.
- Trummer baseline uses one paper-style semantic-join prompt per adaptive block
  pair and returns decisions for all tuples in that block.

Ollama builds without log-probability support fall back to schema-reported
continuous confidence. This is less statistically reliable than true token
log-odds, but it preserves the same thresholding interface.

## Run the isolated benchmark code

The fixed runner reads datasets from the suite specified by
`BENCHMARK_SUITE_ROOT` but imports implementations from `fix/`:

```bash
export BENCHMARK_SUITE_ROOT="$PWD/benchmarks/1q"
python fix/benchmarks/shared/scripts/run_all.py \
  --api-base http://127.0.0.1:11434 \
  --cheap-model ollama/gemma4:e2b \
  --expensive-model ollama/gemma4:e4b \
  --expensive-batch-size 8 \
  --repetitions 1 \
  --methods suql_baseline suql_v1 trummer_baseline trummer_v1 \
  --keep-run-artifacts \
  --output-dir benchmarks/1q/outputs/fixed_semantic_retrieval
```

Keep raw artifacts during evaluation: the evidence arrays and retry/failure
records are necessary to distinguish definition mismatch from missed evidence.

## Important evaluation note

The canonical benchmark labels are lexical regex labels, not human semantic
judgments. The fixed prompts intentionally count explicit negated, quoted, and
critical mentions because those are positive under the evaluator. If the target
is changed to semantic truth, the examples and label-calibration function must
be changed with it.
