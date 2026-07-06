# IMDb semantic-join baseline

This directory reproduces the IMDb review experiment from Immanuel Trummer's
paper *Implementing Semantic Join Operators Efficiently* and its official
implementation:

https://github.com/itrummer/llmjoins

## Benchmark

The source dataset is `data/all_reviews.csv`, copied verbatim from the official
repository's `testdata/all_reviews.csv`. It contains 100 reviews and a
`sentiment` column with `neg` or `pos`.

The paper's preparation procedure is reproduced exactly:

1. Tokenize each review with the GPT-4o tokenizer.
2. Truncate reviews longer than 100 tokens.
3. Use reviews 0-49 as the left table.
4. Use reviews 50-99 as the right table.
5. Join two reviews when:

   ```text
   both reviews are positive or both are negative
   ```

6. Generate ground truth using:

   ```python
   joins = sentiment_left == sentiment_right
   ```

This produces 2,500 labeled candidate pairs.

## Setup

```bash
cd "/Users/annremizova/Desktop/lab m2/project Trummer/baseline"
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r ../requirements.txt
python3 prepare_data.py
```

## Verification without an LLM

The oracle backend uses the sentiment labels as predictions. It verifies the
data preparation, operators, result serialization, and metric calculations:

```bash
python3 run_experiment.py --operator block --backend oracle
```

Expected quality:

```text
precision: 1.0
recall: 1.0
f1: 1.0
```

## Run with Ollama

```bash
python3 run_experiment.py \
  --operator block \
  --backend llm \
  --api-base http://127.0.0.1:11434 \
  --model ollama/gemma2:2b
```

Adaptive join, initialized as in the paper at actual selectivity divided by
100 (`0.5 / 100 = 0.005`):

```bash
python3 run_experiment.py \
  --operator adaptive \
  --backend llm \
  --api-base http://127.0.0.1:11434 \
  --model ollama/gemma2:2b \
  --initial-selectivity 0.005
```

Tuple nested-loop baseline:

```bash
python3 run_experiment.py \
  --operator tuple \
  --backend llm \
  --api-base http://127.0.0.1:11434 \
  --model ollama/gemma2:2b
```

The tuple operator makes 2,500 LLM calls.

## Outputs

Each operator writes:

```text
outputs/<operator>_results.csv
outputs/<operator>_stats.csv
outputs/<operator>_metrics.json
```

Metrics include:

- precision, recall, and F1
- true positives, false positives, and false negatives
- prompts
- input and output tokens
- accumulated LLM-call time
- historical GPT-4 cost using the paper's $0.03/1K input and $0.06/1K output prices

## Differences from the published run

The paper used `gpt-4-0613`, OpenAI client 1.12, a 20-second request timeout,
temperature zero, and a 2,000-token context limit. The runner defaults to the
same context limit and temperature but allows a local Ollama model. Results
from Gemma2 are therefore a reproduction of the workload and algorithms, not
an exact reproduction of GPT-4 accuracy or latency.
