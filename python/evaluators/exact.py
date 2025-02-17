from evaluators.types import EvaluatorResult
from evaluators.utils import _run_evaluator, _arun_evaluator

import json
from typing import Any


def _scorer(inputs: Any, outputs: Any) -> bool:
    if inputs is None or outputs is None:
        raise ValueError("Exact match requires both inputs and outputs")
    # Convert both to JSON strings for deep comparison
    inputs_json = json.dumps(inputs, sort_keys=True)
    outputs_json = json.dumps(outputs, sort_keys=True)
    return inputs_json == outputs_json


def exact_match(*, inputs: Any, outputs: Any, **kwargs: Any) -> EvaluatorResult:
    """
    Performs exact matching between input and output values.

    Args:
        inputs (Any): Inputs to compare
        outputs (Any): Outputs to compare

    Returns:
        EvaluatorResult: Contains match result
    """

    def get_score():
        return _scorer(inputs, outputs)

    return _run_evaluator(
        run_name="exact_match", scorer=get_score, feedback_key="exact_match"
    )


async def exact_match_async(
    *, inputs: Any, outputs: Any, **kwargs: Any
) -> EvaluatorResult:
    """
    Performs exact matching between input and output values.

    Args:
        inputs (Any): Inputs to compare
        outputs (Any): Outputs to compare

    Returns:
        EvaluatorResult: Contains match result
    """

    async def get_score():
        return _scorer(inputs, outputs)

    return await _arun_evaluator(
        run_name="exact_match", scorer=get_score, feedback_key="exact_match"
    )
