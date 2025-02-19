from openevals.types import EvaluatorResult
from openevals.utils import _run_evaluator, _arun_evaluator

import json
from typing import Any


def _scorer(outputs: Any, reference_outputs: Any) -> bool:
    if outputs is None or reference_outputs is None:
        raise ValueError("Exact match requires both outputs and reference_outputs")
    # Convert both to JSON strings for deep comparison
    outputs_json = json.dumps(outputs, sort_keys=True)
    reference_outputs_json = json.dumps(reference_outputs, sort_keys=True)
    return outputs_json == reference_outputs_json


def exact_match(
    *, outputs: Any, reference_outputs: Any, **kwargs: Any
) -> EvaluatorResult:
    """
    Performs exact matching between input and reference output values.

    Args:
        outputs (Any): Outputs to compare
        reference_outputs (Any): Reference outputs to compare

    Returns:
        EvaluatorResult: Contains match result
    """

    def get_score():
        return _scorer(outputs, reference_outputs)

    res = _run_evaluator(
        run_name="exact_match", scorer=get_score, feedback_key="exact_match"
    )
    if isinstance(res, list):
        return res[0]
    return res


async def exact_match_async(
    *, outputs: Any, reference_outputs: Any, **kwargs: Any
) -> EvaluatorResult:
    """
    Performs exact matching between input and reference output values.

    Args:
        outputs (Any): Outputs to compare
        reference_outputs (Any): Reference outputs to compare

    Returns:
        EvaluatorResult: Contains match result
    """

    async def get_score():
        return _scorer(outputs, reference_outputs)

    res = await _arun_evaluator(
        run_name="exact_match", scorer=get_score, feedback_key="exact_match"
    )
    if isinstance(res, list):
        return res[0]
    return res
