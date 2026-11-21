"""Validate synthetic corpus, create/verify the LangSmith dataset; idempotent upload."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATASET_DEFAULT = "companyscope-evidence-v1"


def load_cases(path=BASE / "cases.json"):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(cases) < 10:
        raise ValueError("At least 10 cases required")
    ids = [x["id"] for x in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if case["reference_outputs"]["status"] not in {"supported", "insufficient_evidence"}:
            raise ValueError(f"Unknown status: {case['id']}")
        if not case["source_ids"]:
            raise ValueError(f"No sources: {case['id']}")
        for source_id in case["source_ids"]:
            if f"[{source_id}]" not in case["inputs"]["evidence"]:
                raise ValueError(f"Missing input excerpt {source_id}: {case['id']}")
        if not case["inputs"]["question"].strip() or not case["reference_outputs"]["answer"].strip():
            raise ValueError(f"Empty prompt/reference: {case['id']}")
    return cases


def get_client():
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
    from langsmith import Client
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY missing: fill .env first")
    return Client()


def upload_dataset(client, cases, name):
    """Do not duplicate examples: a nonempty dataset must contain exactly our IDs."""
    try:
        dataset = client.read_dataset(dataset_name=name)
        existing = list(client.list_examples(dataset_id=dataset.id))
        existing_ids = {e.inputs.get("case_id") for e in existing}
        desired_ids = {c["id"] for c in cases}
        if existing and existing_ids != desired_ids:
            raise ValueError(f"Dataset {name!r} exists with different examples; use a new LANGSMITH_DATASET name")
        if existing and len(existing) != len(cases):
            raise ValueError(f"Dataset {name!r} contains duplicates; inspect manually")
        if existing:
            print(f"Dataset already populated: {dataset.id}, {len(existing)} examples")
            return dataset
    except Exception as error:
        # Do not suppress auth/network errors or mismatched existing datasets.
        from langsmith.utils import LangSmithNotFoundError
        if not isinstance(error, LangSmithNotFoundError):
            raise
        dataset = client.create_dataset(dataset_name=name,
            description="CompanyScope synthetic evidence-grounded company research v1; 18 human-labeled cases")
    payload = [{"inputs": x["inputs"], "reference_outputs": x["reference_outputs"],
                "metadata": {"case_id": x["id"],"category": x["category"],
                             "difficulty": x["difficulty"],"source_ids": x["source_ids"]}}
               for x in cases]
    client.create_examples(dataset_id=dataset.id, examples=payload)
    count = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
    if count != len(cases):
        raise RuntimeError(f"Uploaded {count} examples, expected {len(cases)}")
    print(f"Uploaded and verified {count} examples to dataset: {name} / {dataset.id}")
    return dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-check", action="store_true", help="Validate local data; no API")
    args = p.parse_args()
    cases = load_cases()
    print(f"Validated {len(cases)} cases: {dict(Counter(c['category'] for c in cases))}")
    if args.local_check:
        return
    client = get_client()
    name = os.environ.get("LANGSMITH_DATASET", DATASET_DEFAULT)
    ds = upload_dataset(client, cases, name)
    print(f"Dataset ID: {ds.id}")

if __name__ == "__main__":
    main()
