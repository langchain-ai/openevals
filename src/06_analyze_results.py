"""
Step 9: Analyze evaluation results.

Loads results/results_latest.csv (produced by 05_run_evaluation.py), joins
it back to the example metadata (difficulty) from data/examples.json, and
computes aggregate + categorical metrics. Writes results/metrics.json and
prints a human-readable summary used to write evaluation_summary.md.
"""

import json
from pathlib import Path

import pandas as pd

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "results_latest.csv"
EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "examples.json"
METRICS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics.json"


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)
    examples = json.loads(EXAMPLES_PATH.read_text())

    difficulty_by_text = {
        e["inputs"]["transaction_text"]: e["metadata"]["difficulty"] for e in examples
    }
    df["difficulty"] = df["inputs.transaction_text"].map(difficulty_by_text)

    n = len(df)
    metrics = {
        "n_examples": n,
        "category_exact_match_rate": round(df["feedback.category_exact_match"].mean(), 3),
        "correctness_rate": round(df["feedback.correctness"].mean(), 3),
        "by_difficulty": {},
        "worst_examples": [],
        "best_examples_sample": [],
    }

    for difficulty, group in df.groupby("difficulty"):
        metrics["by_difficulty"][difficulty] = {
            "n": len(group),
            "category_exact_match_rate": round(group["feedback.category_exact_match"].mean(), 3),
            "correctness_rate": round(group["feedback.correctness"].mean(), 3),
        }

    failing = df[df["feedback.correctness"] == False]  # noqa: E712
    for _, row in failing.iterrows():
        metrics["worst_examples"].append(
            {
                "transaction_text": row["inputs.transaction_text"],
                "predicted_category": row["outputs.category"],
                "expected_category": row["reference.category"],
                "predicted_merchant": row["outputs.merchant"],
                "expected_merchant": row["reference.merchant"],
                "category_exact_match": bool(row["feedback.category_exact_match"]),
            }
        )

    passing = df[df["feedback.correctness"] == True].head(3)  # noqa: E712
    for _, row in passing.iterrows():
        metrics["best_examples_sample"].append(
            {
                "transaction_text": row["inputs.transaction_text"],
                "predicted_category": row["outputs.category"],
                "predicted_merchant": row["outputs.merchant"],
            }
        )

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("=== Evaluation Summary ===")
    print(f"Examples: {metrics['n_examples']}")
    print(f"Category exact-match rate: {metrics['category_exact_match_rate']:.0%}")
    print(f"LLM-judged correctness rate: {metrics['correctness_rate']:.0%}")
    print("\nBy difficulty:")
    for difficulty, stats in metrics["by_difficulty"].items():
        print(
            f"  {difficulty:8s} (n={stats['n']:2d})  "
            f"category_exact_match={stats['category_exact_match_rate']:.0%}  "
            f"correctness={stats['correctness_rate']:.0%}"
        )
    print(f"\nFailing examples ({len(metrics['worst_examples'])}):")
    for ex in metrics["worst_examples"]:
        print(
            f"  '{ex['transaction_text']}' -> "
            f"predicted=({ex['predicted_category']}, {ex['predicted_merchant']}) "
            f"expected=({ex['expected_category']}, {ex['expected_merchant']})"
        )
    print(f"\nWrote metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
