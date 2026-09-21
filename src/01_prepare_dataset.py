"""
Step 1: Extract and structure examples from the source documents.

Reads data/source_transactions.txt (raw bank-export lines + ground-truth
labels) and writes data/examples.json in the input/output/metadata shape
LangSmith expects.
"""

import json
from pathlib import Path

SOURCE_PATH = Path(__file__).resolve().parent.parent / "data" / "source_transactions.txt"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "examples.json"


def parse_line(line: str) -> dict:
    raw_text, amount, currency, category, merchant, difficulty = line.split("|")
    return {
        "inputs": {
            "transaction_text": raw_text.strip(),
            "amount": float(amount),
            "currency": currency.strip(),
        },
        "outputs": {
            "category": category.strip(),
            "merchant": merchant.strip(),
        },
        "metadata": {
            "difficulty": difficulty.strip(),
            "source": "synthetic_bank_export",
        },
    }


def main() -> None:
    lines = [
        line.strip()
        for line in SOURCE_PATH.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    examples = [parse_line(line) for line in lines]

    for i, example in enumerate(examples, start=1):
        example["metadata"]["id"] = f"txn-{i:02d}"

    if len(examples) < 10:
        raise ValueError(f"Need at least 10 examples, found {len(examples)}")

    OUTPUT_PATH.write_text(json.dumps(examples, indent=2))
    print(f"Parsed {len(examples)} examples from {SOURCE_PATH.name} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
