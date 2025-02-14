from evaluators.types import ChatCompletionMessage, EvaluatorResult
from evaluators.utils import _run_evaluator
from evaluators.trajectory.utils import _is_trajectory_superset

from typing import Any, Union


def trajectory_unordered_match(
    *,
    outputs: Union[list[ChatCompletionMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], dict],
    **kwargs: Any,
) -> EvaluatorResult:
    """
    Evaluate whether an input agent trajectory and called tools contains all the tools used in a reference trajectory.
    This accounts for some differences in an LLM's reasoning process in a case-by-case basis.

    Args:
        outputs (list[ChatCompletionMessage]): Actual trajectory the agent followed
        reference_outputs (list[ChatCompletionMessage]): Ideal reference trajectory the agent should have followed

    Returns:
        EvaluatorResult: Contains a score of 1.0 if trajectory matches, 0.0 otherwise
    """

    def get_score():
        if outputs is None or reference_outputs is None:
            raise ValueError(
                "Trajectory unordered match requires both outputs and reference_outputs"
            )
        unordered_match = _is_trajectory_superset(
            outputs, reference_outputs
        ) and _is_trajectory_superset(reference_outputs, outputs)
        return 1.0 if unordered_match else 0.0

    return _run_evaluator(
        run_name="trajectory_unordered_match",
        scorer=get_score,
        feedback_key="trajectory_unordered_match",
    )
