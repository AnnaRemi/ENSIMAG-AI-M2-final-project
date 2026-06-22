# Common benchmark comparison

Question: Which movies released in 1998 have reviews expressing an overall negative, critical, or strongly unfavorable opinion of the movie?

| implementation | mode | cpu_seconds | engine_seconds | llm_calls | final_answer_rows | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| suql_baseline | llm | 0.1209 | 605.8178 | 12 | 6 | 1.0000 | 1.0000 | 1.0000 |
| trummer_heterogen_v1 | llm | 0.7851 | 597.8642 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 |

Precision and recall use unique movie IDs. `final_answer_rows` retains each implementation's raw output-row count.
CPU time is the benchmark client process CPU time; external Ollama server CPU/GPU time is not included.
