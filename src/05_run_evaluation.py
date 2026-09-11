"""
Step 7: Run the LangSmith evaluation experiment.

Wires the target function, dataset, and evaluators together via
client.evaluate(), then exports the results to results/ for offline
analysis in 06_analyze_results.py.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASET_NAME, EXPERIMENT_PREFIX, MODEL_NAME, get_langsmith_client  # noqa: E402
from importlib import import_module  # noqa: E402

target_module = import_module("03_target_function")
evaluators_module = import_module("04_evaluators")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    client = get_langsmith_client()

    prefix = f"{EXPERIMENT_PREFIX}-{MODEL_NAME}"
    print(f"Running evaluation '{prefix}' on dataset '{DATASET_NAME}'...")

    results = client.evaluate(
        target_module.categorize_transaction,
        data=DATASET_NAME,
        evaluators=[
            evaluators_module.category_exact_match,
            evaluators_module.correctness_evaluator,
        ],
        experiment_prefix=prefix,
        max_concurrency=3,
        metadata={"model": MODEL_NAME, "temperature": 0},
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    df = results.to_pandas()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = RESULTS_DIR / f"results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    latest_path = RESULTS_DIR / "results_latest.csv"
    df.to_csv(latest_path, index=False)

    print(f"Saved {len(df)} rows to {csv_path} and {latest_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
