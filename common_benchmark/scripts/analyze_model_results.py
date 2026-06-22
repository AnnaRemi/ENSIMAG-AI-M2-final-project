#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "suql_baseline": "SUQL",
    "trummer_heterogen_v1": "Trummer",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze all common-benchmark model results.")
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--output-dir",
        help="Defaults to <outputs-dir>/model_metric_plots.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else outputs_dir / "model_metric_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for comparison_path in sorted(outputs_dir.glob("*/comparison.csv")):
        frame = pd.read_csv(comparison_path)
        if set(frame["implementation"]) != set(LABELS):
            continue
        frame["approach"] = frame["implementation"].map(LABELS)
        frame["model_label"] = frame["model"].str.removeprefix("ollama/")
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No complete comparison files found under {outputs_dir}")

    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(output_dir / "analysis_all_results.csv", index=False)

    paired_rows = []
    for model, group in all_results.groupby("model_label", sort=True):
        indexed = group.set_index("approach")
        suql = indexed.loc["SUQL"]
        trummer = indexed.loc["Trummer"]
        paired_rows.append(
            {
                "model": model,
                "suql_engine_seconds": suql["engine_seconds"],
                "trummer_engine_seconds": trummer["engine_seconds"],
                "trummer_time_vs_suql": trummer["engine_seconds"] / suql["engine_seconds"],
                "suql_llm_calls": int(suql["llm_calls"]),
                "trummer_llm_calls": int(trummer["llm_calls"]),
                "suql_rows": int(suql["final_answer_rows"]),
                "trummer_rows": int(trummer["final_answer_rows"]),
                "suql_precision": suql["precision"],
                "trummer_precision": trummer["precision"],
                "suql_recall": suql["recall"],
                "trummer_recall": trummer["recall"],
                "suql_f1": suql["f1"],
                "trummer_f1": trummer["f1"],
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_model_analysis.csv", index=False)

    aggregate = (
        all_results.groupby("approach")
        .agg(
            models=("model_label", "count"),
            nonzero_result_models=("final_answer_rows", lambda values: int((values > 0).sum())),
            mean_precision=("precision", "mean"),
            median_precision=("precision", "median"),
            mean_recall=("recall", "mean"),
            median_recall=("recall", "median"),
            mean_f1=("f1", "mean"),
            median_f1=("f1", "median"),
            perfect_f1_models=("f1", lambda values: int((values == 1).sum())),
            mean_llm_calls=("llm_calls", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "approach_summary.csv", index=False)
    write_report(all_results, paired, aggregate, output_dir / "analysis.md")
    print(f"Wrote analysis to {output_dir / 'analysis.md'}")


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return lines


def write_report(
    all_results: pd.DataFrame,
    paired: pd.DataFrame,
    aggregate: pd.DataFrame,
    path: Path,
) -> None:
    suql = aggregate.set_index("approach").loc["SUQL"]
    trummer = aggregate.set_index("approach").loc["Trummer"]
    trummer_nonzero = paired[paired["trummer_rows"] > 0]
    trummer_faster = paired[paired["trummer_time_vs_suql"] < 1]
    useful_trummer_faster = trummer_nonzero[trummer_nonzero["trummer_time_vs_suql"] < 1]

    paired_table = []
    for row in paired.itertuples(index=False):
        paired_table.append(
            [
                row.model,
                fmt(row.suql_engine_seconds, 1),
                fmt(row.trummer_engine_seconds, 1),
                fmt(row.trummer_time_vs_suql, 2) + "x",
                f"{row.suql_llm_calls}/{row.trummer_llm_calls}",
                f"{row.suql_rows}/{row.trummer_rows}",
                fmt(row.suql_f1),
                fmt(row.trummer_f1),
            ]
        )

    lines = [
        "# SUQL baseline vs Trummer: analysis across models",
        "",
        "## Scope",
        "",
        "Seven Ollama model experiments were evaluated on one fixed question with six ground-truth movie IDs. "
        "Each model used the same 12 semantic candidates after the year filter.",
        "",
        "The `llama3.2` and `phi4-mini` experiments were run locally on the Mac; the other experiments were run on Aker. "
        "Therefore, compare SUQL against Trummer within each model, but do not interpret timing differences between models as pure model-speed differences.",
        "",
        "## Per-model results",
        "",
        *markdown_table(
            [
                "Model",
                "SUQL time (s)",
                "Trummer time (s)",
                "Trummer/SUQL",
                "Calls S/T",
                "Rows S/T",
                "SUQL F1",
                "Trummer F1",
            ],
            paired_table,
        ),
        "",
        "## Aggregate reliability",
        "",
        *markdown_table(
            ["Approach", "Non-zero runs", "Mean precision", "Mean recall", "Mean F1", "Median F1", "Perfect runs"],
            [
                [
                    "SUQL",
                    f"{int(suql.nonzero_result_models)}/{int(suql.models)}",
                    fmt(suql.mean_precision),
                    fmt(suql.mean_recall),
                    fmt(suql.mean_f1),
                    fmt(suql.median_f1),
                    int(suql.perfect_f1_models),
                ],
                [
                    "Trummer",
                    f"{int(trummer.nonzero_result_models)}/{int(trummer.models)}",
                    fmt(trummer.mean_precision),
                    fmt(trummer.mean_recall),
                    fmt(trummer.mean_f1),
                    fmt(trummer.median_f1),
                    int(trummer.perfect_f1_models),
                ],
            ],
        ),
        "",
        "## Findings",
        "",
        f"- SUQL returned non-empty answers for all {int(suql.models)} models. Trummer returned non-empty answers for only "
        f"{int(trummer.nonzero_result_models)}: `llama3.2` and `mistral:7b`.",
        f"- SUQL mean F1 was {fmt(suql.mean_f1)} and median F1 was {fmt(suql.median_f1)}. "
        f"Trummer mean F1 was {fmt(trummer.mean_f1)} and median F1 was {fmt(trummer.median_f1)}.",
        "- SUQL achieved perfect precision and recall with `mistral:7b`, `qwen2.5:3b`, and `qwen2.5:7b`.",
        "- Trummer's best result was `llama3.2` with F1 0.706, but it returned 11 rows for six true movies, producing five false positives.",
        "- Trummer with `mistral:7b` returned seven rows but only three true positives, giving F1 0.462.",
        "- Trummer reduced model requests from 12 to 1 for every model. This is a 12x request-count reduction, but not necessarily a token, energy, or monetary-cost reduction because its single request contains all movie and review rows.",
        f"- Trummer was faster in {len(trummer_faster)}/{len(paired)} paired runs. However, it produced zero rows in "
        f"{len(trummer_faster) - len(useful_trummer_faster)} of those faster runs. It had no paired run that was both faster than SUQL and returned a non-empty answer.",
        "- Client-process CPU time is not useful for judging inference cost because Ollama executes in a separate process. Engine/wall time and model-server telemetry are the relevant measures.",
        "",
        "## Which approach is better?",
        "",
        "### Use SUQL when",
        "",
        "- answer correctness, stable recall, or predictable behavior matters;",
        "- the chosen model has not been explicitly validated with Trummer's strict index-pair output format;",
        "- false positives are costly;",
        "- the structured filter leaves a small or moderate candidate set, as in this benchmark's 12 rows;",
        "- results must remain usable across different local models.",
        "",
        "For the current benchmark, SUQL is the clearly better production choice. It dominates Trummer on quality for every tested model and is not consistently slower.",
        "",
        "### Consider Trummer when",
        "",
        "- minimizing the number of API round trips is more important than retrieval accuracy;",
        "- the model is known to follow the exact `x,y` pair format;",
        "- prompts can batch many candidates without exceeding context limits;",
        "- false positives can be verified by a later stage;",
        "- a calibrated fallback reruns malformed or empty outputs using tuple-level classification.",
        "",
        "The current Trummer implementation is experimental rather than robust. Before using it for a larger benchmark, it should persist raw responses, validate output format, retry malformed outputs, and fall back to smaller blocks or per-row classification.",
        "",
        "## Important limitations",
        "",
        "- This is one question and one run per model; it does not establish general statistical significance.",
        "- Local and Aker hardware timings are mixed. Only within-model SUQL-vs-Trummer timing comparisons are defensible.",
        "- Trummer's zero-row outcomes may be formatting/parser failures rather than genuine semantic decisions because raw model responses were not saved.",
        "- LLM call count alone does not measure total token traffic. SUQL sends 12 smaller prompts; Trummer sends one roughly 3,000-token prompt.",
        "- The benchmark uses one annotated review per movie, not all reviews for each movie.",
        "",
        "## Recommended next experiment",
        "",
        "Run at least five repetitions per model and save raw Trummer responses plus prompt/completion tokens for both approaches. "
        "Then compare median latency, success rate, total tokens, precision, recall, and F1. Add a Trummer retry/fallback path and report its total cost, not only the initial one-call path.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
