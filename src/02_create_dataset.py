"""
Step 2: Create (or reuse) the LangSmith dataset and upload examples.

Idempotent: if a dataset with DATASET_NAME already exists, examples are
upserted into it rather than creating a duplicate dataset.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASET_DESCRIPTION, DATASET_NAME, get_langsmith_client  # noqa: E402

EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "examples.json"


def main() -> None:
    client = get_langsmith_client()
    examples = json.loads(EXAMPLES_PATH.read_text())

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' already exists (id={dataset.id}); reusing it.")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESCRIPTION,
        )
        print(f"Created dataset '{DATASET_NAME}' (id={dataset.id}).")

    response = client.create_examples(
        dataset_id=dataset.id,
        examples=examples,
    )
    print(f"Upserted {len(examples)} examples into dataset '{DATASET_NAME}'.")
    print(f"Response: {response}")

    final_count = len(list(client.list_examples(dataset_id=dataset.id)))
    print(f"Dataset now contains {final_count} examples total.")
    print(f"View it at: {client.api_url.replace('api.smith', 'smith').rstrip('/')}"
          f"/datasets/{dataset.id}")


if __name__ == "__main__":
    main()
