from evaluators.types import EvaluatorResult
from evaluators.utils import _run_evaluator

import json
from typing import Any


def exact_match(*, inputs: Any, outputs: Any, **kwargs: Any) -> EvaluatorResult:
    """
    Performs exact matching between input and output values.

    Args:
        inputs (Any): Inputs to compare
        outputs (Any): Outputs to compare

    Returns:
        MatchResult: Contains match result with score 1.0 for exact match, 0.0 otherwise
    """

    def get_score():
        if inputs is None or outputs is None:
            raise ValueError("Exact match requires both inputs and outputs")
        # Convert both to JSON strings for deep comparison
        inputs_json = json.dumps(inputs, sort_keys=True)
        outputs_json = json.dumps(outputs, sort_keys=True)
        return 1.0 if inputs_json == outputs_json else 0.0

    return _run_evaluator(
        run_name="exact_match", evaluator_fn=get_score, feedback_key="exact_match"
    )
