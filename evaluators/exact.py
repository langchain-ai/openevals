from evaluators.types import EvaluatorResult
import json
from typing import Any


def exact_match(*, inputs: Any, outputs: Any) -> EvaluatorResult:
    """
    Performs exact matching between input and output values.

    Args:
        inputs (Any): Inputs to compare
        outputs (Any): Outputs to compare

    Returns:
        MatchResult: Contains match result with score 1.0 for exact match, 0.0 otherwise
    """
    # Convert both to JSON strings for deep comparison
    inputs_json = json.dumps(inputs, sort_keys=True)
    outputs_json = json.dumps(outputs, sort_keys=True)
    return EvaluatorResult(
        key="exact_match", score=1.0 if inputs_json == outputs_json else 0.0
    )
