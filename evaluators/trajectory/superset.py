from __future__ import annotations
from evaluators.types import ChatCompletionMessage, EvaluatorResult
from evaluators.utils import _run_evaluator, _normalize_to_openai_messages_list
from evaluators.trajectory.utils import _is_trajectory_superset

from typing import Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def trajectory_superset(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    **kwargs: Any,
) -> EvaluatorResult:
    """
    Evaluate whether an agent trajectory and called tools is a superset of a reference trajectory and called tools.
    This means the agent called a superset of the tools specified in the reference trajectory.

    Args:
        outputs (list[ChatCompletionMessage]): Actual trajectory the agent followed
        reference_outputs (list[ChatCompletionMessage]): Ideal reference trajectory the agent should have followed

    Returns:
        EvaluatorResult: Contains a score of 1.0 if trajectory (including called tools) matches, 0.0 otherwise
    """
    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)

    def get_score():
        if outputs is None or reference_outputs is None:
            raise ValueError(
                "Trajectory superset match requires both outputs and reference_outputs"
            )
        is_superset = _is_trajectory_superset(outputs, reference_outputs)
        return 1.0 if is_superset else 0.0

    return _run_evaluator(
        run_name="trajectory_superset",
        scorer=get_score,
        feedback_key="trajectory_superset",
    )
