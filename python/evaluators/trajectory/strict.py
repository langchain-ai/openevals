from __future__ import annotations
from evaluators.types import ChatCompletionMessage, EvaluatorResult
from evaluators.utils import (
    _run_evaluator,
    _arun_evaluator,
    _normalize_to_openai_messages_list,
)

from typing import Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def _scorer(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
) -> float:
    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)
    if outputs is None or reference_outputs is None:
        raise ValueError(
            "Strict trajectory match requires both outputs and reference_outputs"
        )
    if len(outputs) != len(reference_outputs):
        return 0.0
    exact_match = True
    for output, reference_output in zip(outputs, reference_outputs):
        if output["role"] != reference_output["role"]:
            exact_match = False
            break
        elif ("tool_calls" in output and output["tool_calls"] is not None) != (
            "tool_calls" in reference_output
            and reference_output["tool_calls"] is not None
        ):
            # One has tool calls while the other doesn't
            exact_match = False
            break
        elif "tool_calls" in output and output["tool_calls"] is not None:
            # Both have tool calls, compare them
            if len(output["tool_calls"]) != len(reference_output["tool_calls"]):
                exact_match = False
                break
            for output_call, reference_call in zip(
                output["tool_calls"], reference_output["tool_calls"]
            ):
                if (
                    output_call["function"]["name"]
                    != reference_call["function"]["name"]
                ):
                    exact_match = False
                    break
    return exact_match


def trajectory_strict_match(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    **kwargs: Any,
) -> EvaluatorResult:
    """
    Evaluate whether an input agent trajectory and called tools strictly matches a reference trajectory.
    This means that at each step, the agent called the same tools in the same order as specified in the reference trajectory.

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
    return _run_evaluator(
        run_name="trajectory_strict_match",
        scorer=_scorer,
        feedback_key="trajectory_strict_match",
        outputs=outputs,
        reference_outputs=reference_outputs,
    )


async def trajectory_strict_match_async(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    **kwargs: Any,
) -> EvaluatorResult:
    """
    Evaluate whether an input agent trajectory and called tools strictly matches a reference trajectory.
    This means that at each step, the agent called the same tools in the same order as specified in the reference trajectory.

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

    async def async_wrapper(**kwargs: Any):
        return _scorer(**kwargs)

    return await _arun_evaluator(
        run_name="trajectory_strict_match",
        scorer=async_wrapper,
        feedback_key="trajectory_strict_match",
        outputs=outputs,
        reference_outputs=reference_outputs,
    )
