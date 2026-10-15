"""
Step 6: Evaluators for the transaction categorization target function.

Two complementary evaluators are used together:

1. category_exact_match - deterministic, code-based. Category is a fixed
   taxonomy, so exact string match is the correct and cheapest measure of
   whether the classification itself is right.
2. correctness (LLM-as-judge) - handles the merchant-name field, where
   "REWE" vs "REWE Markt" are both reasonable and a strict string match
   would unfairly penalize valid normalizations. An LLM judge compares the
   full output dict against the reference and scores overall correctness.
"""

import sys
from pathlib import Path

from openevals.llm import create_llm_as_judge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JUDGE_MODEL_NAME  # noqa: E402


def category_exact_match(outputs: dict, reference_outputs: dict) -> dict:
    score = float(outputs.get("category") == reference_outputs.get("category"))
    return {
        "key": "category_exact_match",
        "score": score,
        "comment": (
            f"predicted='{outputs.get('category')}' "
            f"expected='{reference_outputs.get('category')}'"
        ),
    }


CORRECTNESS_PROMPT = """You are grading a financial transaction categorization system.

Given the raw bank transaction (<input>), the system's prediction (<output>: category and \
merchant), and the analyst's ground truth (<reference_outputs>), decide whether the \
prediction is correct.

Grading rules:
- The "category" field must match the reference category exactly to be correct.
- The "merchant" field does not need to match character-for-character: accept reasonable \
normalizations, abbreviations, or legal-entity variations of the same real-world merchant \
(e.g. "REWE" and "REWE Markt" are both correct; "Deutsche Bahn" and "DB Vertrieb GmbH" are \
both correct since they refer to the same company). Only mark it wrong if it names a \
different or incorrect entity.
- If either field is clearly wrong, the overall answer is not correct.

Respond with a binary judgment: correct or incorrect.

<input>
{inputs}
</input>

<output>
{outputs}
</output>

<reference_outputs>
{reference_outputs}
</reference_outputs>"""

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model=f"openai:{JUDGE_MODEL_NAME}",
    continuous=False,
)


if __name__ == "__main__":
    sample_inputs = {
        "transaction_text": "SPOTIFY AB STOCKHOLM SWE",
        "amount": -9.99,
        "currency": "EUR",
    }
    sample_outputs = {"category": "Subscriptions", "merchant": "Spotify AB"}
    sample_reference = {"category": "Subscriptions", "merchant": "Spotify"}

    print(category_exact_match(sample_outputs, sample_reference))
    print(
        correctness_evaluator(
            inputs=sample_inputs,
            outputs=sample_outputs,
            reference_outputs=sample_reference,
        )
    )
