#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
        time_ratio = (
            trummer["engine_seconds"] / suql["engine_seconds"]
            if suql["engine_seconds"] > 0
            else float("nan")
        )
        paired_rows.append(
            {
                "model": model,
                "suql_engine_seconds": suql["engine_seconds"],
                "trummer_engine_seconds": trummer["engine_seconds"],
                "trummer_time_vs_suql": time_ratio,
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
    benchmark = json.loads((ROOT / "benchmark.json").read_text())
    write_report(all_results, paired, aggregate, benchmark, output_dir / "analysis.md")
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
    benchmark: dict,
    path: Path,
) -> None:
    suql = aggregate.set_index("approach").loc["SUQL"]
    trummer = aggregate.set_index("approach").loc["Trummer"]
    trummer_nonzero = paired[paired["trummer_rows"] > 0]
    trummer_faster = paired[paired["trummer_time_vs_suql"] < 1]
    useful_trummer_faster = trummer_nonzero[trummer_nonzero["trummer_time_vs_suql"] < 1]
    best_suql = paired.loc[paired["suql_f1"].idxmax()]
    best_trummer = paired.loc[paired["trummer_f1"].idxmax()]

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
        f"{len(paired)} Ollama model experiments were evaluated on one fixed question with "
        f"{benchmark['ground_truth_count']} ground-truth movie IDs.",
        "",
        f"The dataset has {benchmark['row_count']} unique movie/review rows across "
        f"{len(benchmark['represented_years'])} years. SUQL applies the year filter before its "
        "semantic calls; Trummer receives every row and evaluates year inside the join predicate.",
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
        f"- SUQL returned non-empty answers for {int(suql.nonzero_result_models)}/{int(suql.models)} models. "
        f"Trummer returned non-empty answers for {int(trummer.nonzero_result_models)}/{int(trummer.models)} models.",
        f"- SUQL mean F1 was {fmt(suql.mean_f1)} and median F1 was {fmt(suql.median_f1)}. "
        f"Trummer mean F1 was {fmt(trummer.mean_f1)} and median F1 was {fmt(trummer.median_f1)}.",
        f"- Best SUQL F1 was {fmt(best_suql.suql_f1)} with `{best_suql.model}`; "
        f"best Trummer F1 was {fmt(best_trummer.trummer_f1)} with `{best_trummer.model}`.",
        f"- Mean LLM calls were {suql.mean_llm_calls:.1f} for SUQL and "
        f"{trummer.mean_llm_calls:.1f} for Trummer.",
        f"- Trummer was faster in {len(trummer_faster)}/{len(paired)} paired runs. However, it produced zero rows in "
        f"{len(trummer_faster) - len(useful_trummer_faster)} of those faster runs.",
        "- Client-process CPU time is not useful for judging inference cost because Ollama executes in a separate process. Engine/wall time and model-server telemetry are the relevant measures.",
        "",
        "## Which approach is better?",
        "",
        "### Use SUQL when",
        "",
        "- answer correctness, stable recall, or predictable behavior matters;",
        "- the chosen model has not been explicitly validated with Trummer's strict index-pair output format;",
        "- false positives are costly;",
        f"- the structured year filter reduces the workload from {benchmark['row_count']} rows "
        f"to {benchmark['candidate_1998_rows']} rows;",
        "- results must remain usable across different local models.",
        "",
        "### Consider Trummer when",
        "",
        "- minimizing the number of API round trips is more important than retrieval accuracy;",
        "- the model is known to follow the exact `x,y` pair format;",
        "- prompts can batch many candidates without exceeding context limits;",
        "- false positives can be verified by a later stage;",
        "- a calibrated fallback reruns malformed or empty outputs using tuple-level classification.",
        "",
        "Interpret the v2 results using total latency, token counts, and output quality together. "
        "Trummer performs a substantially harder join because the year condition remains inside its LLM predicate.",
        "",
        "## Important limitations",
        "",
        "- This is one question and one run per model; it does not establish general statistical significance.",
        "- Compare SUQL and Trummer within each model and hardware run; cross-machine timing is not directly comparable.",
        "- Trummer's zero-row outcomes may be formatting/parser failures rather than genuine semantic decisions because raw model responses were not saved.",
        "- LLM call count alone does not measure total token traffic; Trummer prompts contain blocks from a 50x50 candidate cross product.",
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
