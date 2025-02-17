from __future__ import annotations
from evaluators.types import ChatCompletionMessage, EvaluatorResult
from evaluators.utils import (
    _run_evaluator,
    _normalize_to_openai_messages_list,
    _arun_evaluator,
)
from evaluators.trajectory.utils import _is_trajectory_superset
from typing import Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def trajectory_subset(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    **kwargs: Any,
) -> EvaluatorResult:
    """
    Evaluate whether an agent trajectory and called tools is a subset of a reference trajectory and called tools.
    This means the agent called a subset of the tools specified in the reference trajectory.

    Args:
        outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Actual trajectory the agent followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.
        reference_outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Ideal reference trajectory the agent should have followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.

    Returns:
        EvaluatorResult: Contains a score of True if trajectory (including called tools) matches, False otherwise
    """
    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)

    def get_score():
        if outputs is None or reference_outputs is None:
            raise ValueError(
                "Trajectory subset match requires both outputs and reference_outputs"
            )
        is_superset = _is_trajectory_superset(reference_outputs, outputs)
        return is_superset

    return _run_evaluator(
        run_name="trajectory_subset",
        scorer=get_score,
        feedback_key="trajectory_subset",
    )


async def trajectory_subset_async(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    **kwargs: Any,
) -> EvaluatorResult:
    """
    Evaluate whether an agent trajectory and called tools is a subset of a reference trajectory and called tools.
    This means the agent called a subset of the tools specified in the reference trajectory.

    Args:
        outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Actual trajectory the agent followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.
        reference_outputs (Union[list[ChatCompletionMessage], list[BaseMessage], dict]): Ideal reference trajectory the agent should have followed.
            May be a list of OpenAI messages, a list of LangChain messages, or a dictionary containing
            a "messages" key with one of the above.

    Returns:
        EvaluatorResult: Contains a score of True if trajectory (including called tools) matches, False otherwise
    """
    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)

    async def aget_score():
        if outputs is None or reference_outputs is None:
            raise ValueError(
                "Trajectory subset match requires both outputs and reference_outputs"
            )
        is_superset = _is_trajectory_superset(reference_outputs, outputs)
        return is_superset

    return await _arun_evaluator(
        run_name="trajectory_subset",
        scorer=aget_score,
        feedback_key="trajectory_subset",
    )
