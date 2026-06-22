#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import tiktoken


def shorten(text: str, encoder, max_tokens: int = 100) -> str:
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens]) + " ..."


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Trummer's IMDb review benchmark.")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    source = data_dir / "all_reviews.csv"
    encoder = tiktoken.encoding_for_model("gpt-4o")

    with source.open(newline="", encoding="utf-8") as handle:
        reviews = list(csv.DictReader(handle))
    if len(reviews) != 100:
        raise ValueError(f"Expected 100 official reviews, found {len(reviews)}")

    prepared = [
        {
            "review_id": str(index),
            "text": shorten(row["text"], encoder),
            "sentiment": row["sentiment"],
        }
        for index, row in enumerate(reviews)
    ]
    left = prepared[:50]
    right = prepared[50:]

    ground_truth = []
    for left_row in left:
        for right_row in right:
            ground_truth.append(
                {
                    "left_id": left_row["review_id"],
                    "right_id": right_row["review_id"],
                    "sentiment_left": left_row["sentiment"],
                    "sentiment_right": right_row["sentiment"],
                    "joins": str(left_row["sentiment"] == right_row["sentiment"]),
                }
            )

    write_csv(data_dir / "reviews_1.csv", left, ["review_id", "text", "sentiment"])
    write_csv(data_dir / "reviews_2.csv", right, ["review_id", "text", "sentiment"])
    write_csv(
        data_dir / "same_reviews_ground_truth.csv",
        ground_truth,
        ["left_id", "right_id", "sentiment_left", "sentiment_right", "joins"],
    )

    positives = sum(row["joins"] == "True" for row in ground_truth)
    print(f"Left reviews: {len(left)}")
    print(f"Right reviews: {len(right)}")
    print(f"Ground-truth pairs: {len(ground_truth)}")
    print(f"Positive pairs: {positives}")
    print(f"Selectivity: {positives / len(ground_truth):.6f}")


if __name__ == "__main__":
    main()
