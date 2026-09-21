"""
Step 3: Target function under evaluation.

Takes a raw bank transaction line and asks an LLM to classify it into a
category and normalize the merchant name. Wrapped in @traceable so every
call is logged to LangSmith, and the underlying OpenAI client is wrapped
via wrap_openai for the same reason.
"""

import json
import sys
from pathlib import Path

from langsmith import traceable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CATEGORIES, MODEL_NAME, get_openai_client  # noqa: E402

EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "examples.json"

_client = get_openai_client()

SYSTEM_PROMPT = f"""You are a financial transaction categorization assistant for a personal \
finance app. You receive a single raw bank transaction line exactly as it comes from a bank \
export (often abbreviated, in German or English, with reference codes mixed in).

Classify it into exactly one of these categories:
{", ".join(CATEGORIES)}

Also extract a clean, human-readable merchant or counterparty name from the raw text \
(strip reference numbers, card terminal codes, and city/country suffixes). If no merchant \
can be identified, use "Unknown".

Respond with ONLY a JSON object: {{"category": "<one category from the list>", \
"merchant": "<clean merchant name>"}}"""


@traceable(run_type="chain", name="categorize_transaction")
def categorize_transaction(inputs: dict) -> dict:
    """
    Args:
        inputs: {"transaction_text": str, "amount": float, "currency": str}

    Returns:
        {"category": str, "merchant": str} (or {"category": "Other",
        "merchant": "Unknown", "error": str} if the call fails)
    """
    user_content = (
        f"Transaction text: {inputs['transaction_text']}\n"
        f"Amount: {inputs['amount']} {inputs['currency']}"
    )

    try:
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        return {
            "category": parsed.get("category", "Other"),
            "merchant": parsed.get("merchant", "Unknown"),
        }
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure -> safe fallback
        return {"category": "Other", "merchant": "Unknown", "error": str(exc)}


if __name__ == "__main__":
    examples = json.loads(EXAMPLES_PATH.read_text())
    for example in examples[:3]:
        result = categorize_transaction(example["inputs"])
        print(f"input:     {example['inputs']}")
        print(f"expected:  {example['outputs']}")
        print(f"predicted: {result}")
        print("-" * 60)
