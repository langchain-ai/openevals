import json
from evaluators.types import EvaluatorResult
from evaluators.utils import _run_evaluator, _arun_evaluator
from typing import Any


def _scorer(inputs: Any, outputs: Any) -> float:
    if inputs is None or outputs is None:
        raise ValueError("Levenshtein distance requires both inputs and outputs")
    if not isinstance(inputs, str):
        inputs = json.dumps(inputs)
    if not isinstance(outputs, str):
        outputs = json.dumps(outputs)
    # Create a matrix of size (m+1)x(n+1) where m and n are the string lengths
    m, n = len(inputs), len(outputs)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if inputs[i - 1] == outputs[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    # Calculate the distance and normalize it to a score between 0 and 1
    distance = dp[m][n]
    max_length = max(m, n)
    score = 1.0 - (distance / max_length) if max_length > 0 else 1.0
    return score


def levenshtein_distance(
    *, inputs: Any, outputs: Any, **kwargs: Any
) -> EvaluatorResult:
    """
    Evaluates the actual output and reference output for similarity by Levenshtein distance.

    Args:
        inputs (Any): Inputs to compare
        outputs (Any): Outputs to compare

    Returns:
        EvaluatorResult: Contains match result with score between 0.0 and 1.0, where 1.0 indicates
        an exact match and lower values indicate greater differences
    """

    def get_score():
        return _scorer(inputs, outputs)

    return _run_evaluator(
        run_name="levenshtein_distance",
        scorer=get_score,
        feedback_key="levenshtein_distance",
    )


async def levenshtein_distance_async(
    *, inputs: Any, outputs: Any, **kwargs: Any
) -> EvaluatorResult:
    """
    Evaluates the actual output and reference output for similarity by Levenshtein distance.

    Args:
        inputs (Any): Inputs to compare
        outputs (Any): Outputs to compare

    Returns:
        EvaluatorResult: Contains match result with score between 0.0 and 1.0, where 1.0 indicates
        an exact match and lower values indicate greater differences
    """

    async def get_score():
        return _scorer(inputs, outputs)

    return await _arun_evaluator(
        run_name="levenshtein_distance",
        scorer=get_score,
        feedback_key="levenshtein_distance",
    )
