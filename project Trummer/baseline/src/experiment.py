from __future__ import annotations

import csv
import json
import math
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken


PREDICATE = "both reviews are positive or both are negative"
PAIR_RE = re.compile(r"(\d+)\s*,\s*(\d+)")


@dataclass
class CallStats:
    operator: str
    round: int
    left_block: int
    right_block: int
    tokens_read: int
    tokens_written: int
    seconds: float
    overflow: bool


class ChatClient:
    def __init__(self, api_base: str, model: str, api_key: str = "", timeout: int = 300):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @property
    def is_ollama(self) -> bool:
        return self.model.startswith("ollama/")

    def chat(self, prompt: str, max_tokens: int, stop: list[str] | None = None) -> dict:
        if self.is_ollama:
            payload = {
                "model": self.model.removeprefix("ollama/"),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            }
            if stop:
                payload["options"]["stop"] = stop
            endpoint = self.api_base + "/api/chat"
            headers = {"Content-Type": "application/json"}
        else:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if stop:
                payload["stop"] = stop
            endpoint = self.api_base + "/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        if self.is_ollama:
            done_reason = str(data.get("done_reason", "stop"))
            return {
                "content": str(data.get("message", {}).get("content", "")),
                "tokens_read": int(data.get("prompt_eval_count", 0) or 0),
                "tokens_written": int(data.get("eval_count", 0) or 0),
                "overflow": done_reason not in {"stop", "unload"},
            }

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return {
            "content": str(choice.get("message", {}).get("content", "")),
            "tokens_read": int(usage.get("prompt_tokens", 0) or 0),
            "tokens_written": int(usage.get("completion_tokens", 0) or 0),
            "overflow": str(choice.get("finish_reason", "")) != "stop",
        }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tuple_prompt(text1: str, text2: str) -> str:
    return "\n".join(
        [
            f'Is the following true ("Yes"/"No"): {PREDICATE}?',
            f"Text 1: {text1}",
            f"Text 2: {text2}",
            "Answer:",
        ]
    )


def block_prompt(left: list[dict], right: list[dict]) -> str:
    parts = [
        "Find indexes x,y where x is the number of an entry in collection 1 "
        f"and y the number of an entry in collection 2 such that {PREDICATE} "
        "(make sure to catch all pairs!)!",
        "Separate index pairs by semicolons.",
        'Write "Finished" after the last pair!',
        "Text Collection 1:",
    ]
    parts.extend(f"{index}: {row['text']}" for index, row in enumerate(left, 1))
    parts.append("Text Collection 2:")
    parts.extend(f"{index}: {row['text']}" for index, row in enumerate(right, 1))
    parts.append("Index pairs:")
    return "\n".join(parts)


def optimal_block_size(
    s1: float,
    s2: float,
    s3: float,
    token_limit: int,
    static_prompt_size: int,
    selectivity: float,
) -> tuple[int, int]:
    estimate = max(selectivity, 0.0000001)
    available = token_limit - static_prompt_size
    numerator = math.sqrt(
        s1 * s1 * s2 * s2 + s1 * s2 * s3 * estimate * available
    ) - s1 * s2
    b1 = max(1, math.floor(numerator / (s1 * s3 * estimate)))
    b2 = max(1, math.floor((available - b1 * s1) / (s2 + b1 * s3 * estimate)))
    return b1, b2


def chunks(rows: list[dict], size: int) -> list[list[dict]]:
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def tuple_join(
    left: list[dict],
    right: list[dict],
    client: ChatClient | None,
    backend: str,
) -> tuple[list[CallStats], set[tuple[str, str]]]:
    stats = []
    results = set()
    total = len(left) * len(right)
    counter = 0
    for left_row in left:
        for right_row in right:
            counter += 1
            started = time.time()
            prompt = tuple_prompt(left_row["text"], right_row["text"])
            if backend == "oracle":
                answer = "Yes" if left_row["sentiment"] == right_row["sentiment"] else "No"
                tokens_read = 0
                tokens_written = 0
            else:
                response = client.chat(prompt, max_tokens=1)
                answer = response["content"]
                tokens_read = response["tokens_read"]
                tokens_written = response["tokens_written"]
            if answer.strip().lower().startswith("yes"):
                results.add((left_row["review_id"], right_row["review_id"]))
            stats.append(
                CallStats(
                    "tuple",
                    1,
                    int(left_row["review_id"]),
                    int(right_row["review_id"]),
                    tokens_read,
                    tokens_written,
                    time.time() - started,
                    False,
                )
            )
            print(f"Tuple pair {counter}/{total}", flush=True)
    return stats, results


def join_blocks(
    left_block: list[dict],
    right_block: list[dict],
    client: ChatClient | None,
    backend: str,
    token_limit: int,
    encoder,
    operator: str,
    round_number: int,
    left_index: int,
    right_index: int,
) -> tuple[CallStats, set[tuple[str, str]]]:
    started = time.time()
    prompt = block_prompt(left_block, right_block)
    prompt_tokens = len(encoder.encode(prompt))
    max_tokens = min(
        token_limit - prompt_tokens,
        max(16, len(left_block) * len(right_block) * 6 + 8),
    )
    if max_tokens < 1:
        return (
            CallStats(operator, round_number, left_index, right_index, 0, 0, time.time() - started, True),
            set(),
        )

    if backend == "oracle":
        pairs = {
            (left_row["review_id"], right_row["review_id"])
            for left_row in left_block
            for right_row in right_block
            if left_row["sentiment"] == right_row["sentiment"]
        }
        response = {
            "tokens_read": 0,
            "tokens_written": 0,
            "overflow": False,
            "content": "",
        }
    else:
        response = client.chat(prompt, max_tokens=max_tokens, stop=["Finished"])
        pairs = set()
        for raw_left, raw_right in PAIR_RE.findall(response["content"]):
            left_pos = int(raw_left) - 1
            right_pos = int(raw_right) - 1
            if 0 <= left_pos < len(left_block) and 0 <= right_pos < len(right_block):
                pairs.add(
                    (
                        left_block[left_pos]["review_id"],
                        right_block[right_pos]["review_id"],
                    )
                )

    stat = CallStats(
        operator,
        round_number,
        left_index,
        right_index,
        response["tokens_read"],
        response["tokens_written"],
        time.time() - started,
        bool(response["overflow"]),
    )
    return stat, pairs


def block_join(
    left: list[dict],
    right: list[dict],
    client: ChatClient | None,
    backend: str,
    token_limit: int,
    selectivity: float,
    encoder,
    operator: str = "block",
    round_number: int = 1,
) -> tuple[list[CallStats], set[tuple[str, str]], bool]:
    s1 = sum(len(encoder.encode(row["text"])) for row in left) / len(left)
    s2 = sum(len(encoder.encode(row["text"])) for row in right) / len(right)
    static_size = len(encoder.encode(block_prompt([], [])))
    b1, b2 = optimal_block_size(s1, s2, 4, token_limit, static_size, selectivity)
    left_blocks = chunks(left, b1)
    right_blocks = chunks(right, b2)
    print(f"Block sizes: left={b1}, right={b2}", flush=True)

    stats = []
    results = set()
    overflow = False
    for left_index, left_block in enumerate(left_blocks, 1):
        if overflow:
            break
        for right_index, right_block in enumerate(right_blocks, 1):
            print(
                f"Joining block {left_index}/{len(left_blocks)} with "
                f"{right_index}/{len(right_blocks)}",
                flush=True,
            )
            stat, pairs = join_blocks(
                left_block,
                right_block,
                client,
                backend,
                token_limit,
                encoder,
                operator,
                round_number,
                left_index,
                right_index,
            )
            stats.append(stat)
            results.update(pairs)
            overflow = stat.overflow
            if overflow:
                break
    return stats, results, overflow


def adaptive_join(
    left: list[dict],
    right: list[dict],
    client: ChatClient | None,
    backend: str,
    token_limit: int,
    initial_selectivity: float,
    encoder,
) -> tuple[list[CallStats], set[tuple[str, str]]]:
    estimate = initial_selectivity
    all_stats = []
    for round_number in range(1, 9):
        stats, results, overflow = block_join(
            left,
            right,
            client,
            backend,
            token_limit,
            estimate,
            encoder,
            operator="adaptive",
            round_number=round_number,
        )
        all_stats.extend(stats)
        if not overflow:
            return all_stats, results
        estimate *= 4
        print(f"Overflow: increasing selectivity estimate to {estimate}", flush=True)
    raise RuntimeError("Adaptive join did not converge after eight rounds")


def load_ground_truth(path: Path) -> set[tuple[str, str]]:
    return {
        (row["left_id"], row["right_id"])
        for row in read_rows(path)
        if row["joins"].lower() == "true"
    }


def evaluate(
    expected: set[tuple[str, str]],
    predicted: set[tuple[str, str]],
    stats: list[CallStats],
) -> dict:
    true_positives = len(expected & predicted)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 0.0 if precision == 0 or recall == 0 else 2 * precision * recall / (precision + recall)
    tokens_read = sum(item.tokens_read for item in stats)
    tokens_written = sum(item.tokens_written for item in stats)
    seconds = sum(item.seconds for item in stats)
    historical_gpt4_usd = tokens_read * 0.03 / 1000 + tokens_written * 0.06 / 1000
    return {
        "ground_truth_pairs": len(expected),
        "predicted_pairs": len(predicted),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prompts": len(stats),
        "tokens_read": tokens_read,
        "tokens_written": tokens_written,
        "seconds": seconds,
        "historical_gpt4_usd": historical_gpt4_usd,
    }


def save_outputs(
    output_dir: Path,
    operator: str,
    results: set[tuple[str, str]],
    stats: list[CallStats],
    metrics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        output_dir / f"{operator}_results.csv",
        [{"left_id": left_id, "right_id": right_id} for left_id, right_id in sorted(results)],
        ["left_id", "right_id"],
    )
    write_rows(
        output_dir / f"{operator}_stats.csv",
        [asdict(item) for item in stats],
        list(CallStats.__dataclass_fields__),
    )
    (output_dir / f"{operator}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
