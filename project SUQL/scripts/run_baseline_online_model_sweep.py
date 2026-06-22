#!/usr/bin/env python3
"""Sweep Ollama models for baseline vs online_join on the 200-row sample.

Each model writes a complete benchmark folder under:

  benchmarks/baseline_vs_join/sample_200_model_sweeps/<model_name>/

The folder contains benchmark_compare.py outputs/logs, metrics.csv,
console.log, and question_vs_metrics.svg.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
DEFAULT_OUTPUT_ROOT = BENCH_DIR / "baseline_vs_join" / "sample_200_model_sweeps"
DEFAULT_SAMPLE_DIR = ROOT / "data_samples" / "data_sample_200"

# Existing Stage_2 sweeps already tried the first five candidates as cheap
# scorers. The qwen/llama 3B variants are useful extra small-model candidates.
DEFAULT_MODELS = [
    "gemma2:2b",
    "llama3.2:1b",
    "smollm2:360m",
    "smollm2:1.7b",
    "tinyllama:1.1b",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "llama3.2:3b",
]


def litellm_model_name(model: str) -> str:
    return model if model.startswith("ollama/") else f"ollama/{model}"


def ollama_model_name(model: str) -> str:
    return model.removeprefix("ollama/")


def safe_model_name(model: str) -> str:
    name = ollama_model_name(model)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def api_base_to_ollama_host(api_base: str) -> str:
    parsed = urlparse(api_base)
    if parsed.netloc:
        return parsed.netloc
    return api_base.removeprefix("http://").removeprefix("https://").rstrip("/")


def find_ollama(explicit: str | None) -> str:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Ollama binary not found: {explicit}")

    for candidate in ("ollama",):
        resolved = subprocess.run(
            ["sh", "-c", f"command -v {candidate}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip()
        if resolved:
            return resolved

    for candidate in (
        Path.home() / ".local" / "ollama" / "bin" / "ollama",
        Path.home() / ".local" / "bin" / "ollama",
        Path.home() / "bin" / "ollama",
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
        Path("/opt/ollama/bin/ollama"),
    ):
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError("Ollama binary not found. Set --ollama-bin or load the Ollama module.")


def installed_models(api_base: str) -> set[str]:
    url = api_base.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc}") from exc

    names: set[str] = set()
    for item in payload.get("models", []):
        name = item.get("name")
        if name:
            names.add(str(name))
    return names


def pull_model(ollama_bin: str, model: str, api_base: str, log_path: Path) -> int:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = api_base_to_ollama_host(api_base)
    proc = subprocess.run(
        [ollama_bin, "pull", ollama_model_name(model)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def as_float(value: object) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_model(model: str, model_dir: Path, status: str, error: str = "") -> dict[str, str]:
    metrics_path = model_dir / "metrics.csv"
    row: dict[str, str] = {
        "model": litellm_model_name(model),
        "ollama_model": ollama_model_name(model),
        "model_dir": str(model_dir.relative_to(ROOT)),
        "status": status,
        "error": error,
        "queries": "0",
        "failed_queries": "",
        "total_wall_seconds": "",
        "mean_wall_seconds": "",
        "mean_engine_seconds": "",
        "total_llm_prompts": "",
        "mean_llm_prompts": "",
        "mean_result_rows": "",
        "plot": "",
    }

    if not metrics_path.exists():
        return row

    rows = read_metrics(metrics_path)
    failed = [item for item in rows if int(as_float(item.get("exit_code", "-1"))) != 0]
    row["queries"] = str(len({item.get("query_id", "") for item in rows}))
    row["failed_queries"] = str(len(failed))
    row["status"] = "failed_queries" if failed else status
    if failed and not error:
        row["error"] = "; ".join(
            f"{item.get('project')}:{item.get('query_id')} exit={item.get('exit_code')}"
            for item in failed[:6]
        )

    if rows:
        wall = [as_float(item.get("wall_seconds", "")) for item in rows]
        engine = [as_float(item.get("engine_seconds", "")) for item in rows]
        prompts = [as_float(item.get("llm_prompts", "")) for item in rows]
        result_rows = [as_float(item.get("result_rows", "")) for item in rows]
        row["total_wall_seconds"] = f"{sum(wall):.6g}"
        row["mean_wall_seconds"] = f"{sum(wall) / len(wall):.6g}"
        row["mean_engine_seconds"] = f"{sum(engine) / len(engine):.6g}"
        row["total_llm_prompts"] = f"{sum(prompts):.6g}"
        row["mean_llm_prompts"] = f"{sum(prompts) / len(prompts):.6g}"
        row["mean_result_rows"] = f"{sum(result_rows) / len(result_rows):.6g}"

    plot_path = model_dir / "question_vs_metrics.svg"
    if plot_path.exists():
        row["plot"] = str(plot_path.relative_to(ROOT))
    return row


def run_benchmark(
    model: str,
    python: str,
    api_base: str,
    sample_dir: Path,
    output_root: Path,
    plot_script: Path,
    title_prefix: str,
) -> dict[str, str]:
    safe_name = safe_model_name(model)
    model_dir = output_root / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    console_log = model_dir / "console.log"
    run_name = str(model_dir.relative_to(BENCH_DIR))

    cmd = [
        python,
        "-u",
        str(ROOT / "benchmark_compare.py"),
        "--sample-size",
        "200",
        "--data-sample-dir",
        str(sample_dir),
        "--api-base",
        api_base,
        "--model",
        litellm_model_name(model),
        "--python",
        python,
        "--run-name",
        run_name,
    ]

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    console_log.write_text(proc.stdout, encoding="utf-8")

    if proc.returncode != 0:
        return summarize_model(
            model,
            model_dir,
            "failed",
            f"benchmark_compare.py exited {proc.returncode}; see {console_log.relative_to(ROOT)}",
        )

    plot_path = model_dir / "question_vs_metrics.svg"
    plot_proc = subprocess.run(
        [
            python,
            str(plot_script),
            "--metrics",
            str(model_dir / "metrics.csv"),
            "--output",
            str(plot_path),
            "--title",
            f"{title_prefix} - {ollama_model_name(model)}, sample 200",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (model_dir / "plot.log").write_text(plot_proc.stdout, encoding="utf-8")
    status = "ok" if plot_proc.returncode == 0 else "plot_failed"
    error = "" if plot_proc.returncode == 0 else f"plot script exited {plot_proc.returncode}; see plot.log"
    summary = summarize_model(model, model_dir, status, error)
    summary["benchmark_seconds"] = f"{elapsed:.2f}"
    return summary


def write_summary(rows: list[dict[str, str]], output_root: Path) -> None:
    if not rows:
        return

    summary_path = output_root / "sweep_summary.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("failed_queries") == "0" and row.get("mean_wall_seconds")
    ]
    if successful:
        best = min(successful, key=lambda row: as_float(row["mean_wall_seconds"]))
        (output_root / "best_model.txt").write_text(
            "Best model by lowest mean wall_seconds among successful runs:\n"
            f"model={best['model']}\n"
            f"mean_wall_seconds={best['mean_wall_seconds']}\n"
            f"total_wall_seconds={best['total_wall_seconds']}\n"
            f"plot={best.get('plot', '')}\n"
            f"model_dir={best['model_dir']}\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Ollama models for baseline vs online_join on sample 200.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--api-base", default=os.environ.get("SUQL_API_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable)
    parser.add_argument("--ollama-bin", help="Path to ollama. Only needed with --pull-models.")
    parser.add_argument("--pull-models", action="store_true", help="Run `ollama pull` for missing models before benchmarking.")
    parser.add_argument("--installed-only", action="store_true", help="Skip models that are not installed and not pulled.")
    parser.add_argument("--allow-phi4", action="store_true", help="Allow phi4 model names in the sweep.")
    parser.add_argument("--title-prefix", default="Baseline vs Online Join")
    args = parser.parse_args()

    sample_joined = args.sample_dir / "imdb_joined.csv"
    if not sample_joined.exists():
        raise SystemExit(f"Missing sample file: {sample_joined}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    plot_script = ROOT / "scripts" / "plot_baseline_online_question_metrics_svg.py"
    if not plot_script.exists():
        raise SystemExit(f"Missing plot script: {plot_script}")

    models = list(dict.fromkeys(args.models))
    if not args.allow_phi4:
        phi4_models = [model for model in models if "phi4" in model.lower()]
        if phi4_models:
            raise SystemExit(
                "This sweep is configured for non-phi4 models. Remove these or pass --allow-phi4: "
                + ", ".join(phi4_models)
            )

    try:
        available = installed_models(args.api_base)
    except RuntimeError as exc:
        if args.pull_models:
            available = set()
        else:
            raise SystemExit(str(exc)) from exc

    ollama_bin = find_ollama(args.ollama_bin) if args.pull_models else ""
    summaries: list[dict[str, str]] = []

    for model in models:
        safe_name = safe_model_name(model)
        model_dir = args.output_root / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        plain_model = ollama_model_name(model)

        if plain_model not in available:
            if args.pull_models:
                print(f"\n=== Pulling {plain_model} ===", flush=True)
                pull_code = pull_model(ollama_bin, model, args.api_base, model_dir / "pull.log")
                if pull_code != 0:
                    summary = summarize_model(
                        model,
                        model_dir,
                        "pull_failed",
                        f"ollama pull exited {pull_code}; see {(model_dir / 'pull.log').relative_to(ROOT)}",
                    )
                    summaries.append(summary)
                    write_summary(summaries, args.output_root)
                    print(f"Skipping {plain_model}: pull failed")
                    continue
                available.add(plain_model)
            elif args.installed_only:
                summary = summarize_model(model, model_dir, "skipped_not_installed", "Model is not installed in Ollama.")
                summaries.append(summary)
                write_summary(summaries, args.output_root)
                print(f"Skipping {plain_model}: not installed")
                continue

        print(f"\n=== Running {plain_model} ===", flush=True)
        summary = run_benchmark(
            model=model,
            python=args.python,
            api_base=args.api_base,
            sample_dir=args.sample_dir,
            output_root=args.output_root,
            plot_script=plot_script,
            title_prefix=args.title_prefix,
        )
        summaries.append(summary)
        write_summary(summaries, args.output_root)
        print(
            f"{plain_model}: status={summary['status']} "
            f"mean_wall={summary.get('mean_wall_seconds', '')} "
            f"plot={summary.get('plot', '')} "
            f"error={summary.get('error', '')}"
        )

    print(f"\nSweep summary: {args.output_root / 'sweep_summary.csv'}")
    best_path = args.output_root / "best_model.txt"
    if best_path.exists():
        print(best_path.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
