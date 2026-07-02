# Cheap block join vs heterogen versions

| version | model | cheap_model | expensive_model | wall_seconds | llm_calls | cheap_calls | expensive_calls | final_answer_rows | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Block join cheap | ollama/qwen3:0.6b | ollama/qwen3:0.6b |  | 11.8400 | 14.0000 | 14.0000 | 0.0000 | 3.0000 | 0.6667 | 0.1538 | 0.2500 |
| Block join expensive | ollama/qwen3:1.7b |  | ollama/qwen3:1.7b | 23.9989 | 14.0000 | 0.0000 | 14.0000 | 5.0000 | 0.4000 | 0.1538 | 0.2222 |
| Row-wise cascade | ollama/qwen3:0.6b->ollama/qwen3:1.7b | ollama/qwen3:0.6b | ollama/qwen3:1.7b | 33.5566 | 60.0000 | 50.0000 | 10.0000 | 40.0000 | 0.3250 | 1.0000 | 0.4906 |
| Structured pruning block join | ollama/qwen3:1.7b |  | ollama/qwen3:1.7b | 11.2340 | 4.0000 | 0.0000 | 4.0000 | 10.0000 | 0.8000 | 0.6154 | 0.6957 |
| Batch-wise cascade | ollama/qwen3:0.6b->ollama/qwen3:1.7b | ollama/qwen3:0.6b | ollama/qwen3:1.7b | 37.6516 | 10.0000 | 7.0000 | 3.0000 | 36.0000 | 0.3333 | 0.9231 | 0.4898 |
| Structured pruning cascade | ollama/qwen3:0.6b->ollama/qwen3:1.7b | ollama/qwen3:0.6b | ollama/qwen3:1.7b | 13.3391 | 28.0000 | 25.0000 | 3.0000 | 24.9091 | 0.5220 | 1.0000 | 0.6859 |

Generated plots:

- `metrics_precision_recall_f1.png`
- `time_bar_plot.png`
- `calls_bar_plot.png`
