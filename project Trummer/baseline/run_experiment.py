#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import tiktoken

from src.experiment import (
    ChatClient,
    adaptive_join,
    block_join,
    evaluate,
    load_ground_truth,
    read_rows,
    save_outputs,
    tuple_join,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Trummer's IMDb semantic-join experiment.")
    parser.add_argument("--operator", choices=["tuple", "block", "adaptive"], default="block")
    parser.add_argument("--backend", choices=["llm", "oracle"], default="llm")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="ollama/gemma2:2b")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--token-limit", type=int, default=2000)
    parser.add_argument("--selectivity", type=float, default=1.0)
    parser.add_argument("--initial-selectivity", type=float, default=0.005)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    required = [
        data_dir / "reviews_1.csv",
        data_dir / "reviews_2.csv",
        data_dir / "same_reviews_ground_truth.csv",
    ]
    if not all(path.exists() for path in required):
        raise FileNotFoundError("Prepared data is missing. Run: python3 prepare_data.py")

    left = read_rows(required[0])
    right = read_rows(required[1])
    expected = load_ground_truth(required[2])
    encoder = tiktoken.encoding_for_model("gpt-4o")
    client = None
    if args.backend == "llm":
        client = ChatClient(args.api_base, args.model, args.api_key, args.timeout)

    if args.operator == "tuple":
        stats, results = tuple_join(left, right, client, args.backend)
    elif args.operator == "block":
        stats, results, overflow = block_join(
            left,
            right,
            client,
            args.backend,
            args.token_limit,
            args.selectivity,
            encoder,
        )
        if overflow:
            raise RuntimeError("Block join overflowed; increase selectivity or use adaptive")
    else:
        stats, results = adaptive_join(
            left,
            right,
            client,
            args.backend,
            args.token_limit,
            args.initial_selectivity,
            encoder,
        )

    metrics = evaluate(expected, results, stats)
    save_outputs(Path(args.output_dir), args.operator, results, stats, metrics)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
