from __future__ import annotations
from warnings import warn

from openevals.trajectory.utils import _normalize_to_openai_messages_list
from openevals.types import (
    ChatCompletionMessage,
    ToolArgsMatchMode,
    ToolArgsMatchOverrides,
)
from openevals.utils import _run_evaluator, _arun_evaluator
from openevals.trajectory.utils import _is_trajectory_superset

from typing import Any, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def _scorer(
    *,
    outputs: list[ChatCompletionMessage],
    reference_outputs: list[ChatCompletionMessage],
    tool_args_match_mode: ToolArgsMatchMode,
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides] = None,
    **kwargs: Any,
):
    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)
    if outputs is None or reference_outputs is None:
        raise ValueError(
            "Strict trajectory match requires both outputs and reference_outputs"
        )
    if len(outputs) != len(reference_outputs):
        return False
    for output, reference_output in zip(outputs, reference_outputs):
        if output["role"] != reference_output["role"]:
            return False
        elif ("tool_calls" in output and output["tool_calls"] is not None) != (
            "tool_calls" in reference_output
            and reference_output["tool_calls"] is not None
        ):
            # One has tool calls while the other doesn't
            return False
        elif "tool_calls" in output and output["tool_calls"] is not None:
            # Both have tool calls, compare them
            if not isinstance(output["tool_calls"], list) or not isinstance(
                reference_output["tool_calls"], list
            ):
                return False
            if len(output["tool_calls"]) != len(reference_output["tool_calls"]):
                return False
            # Same number of calls, so a superset is a one-to-one match
            if not _is_trajectory_superset(
                [output],
                [reference_output],
                tool_args_match_mode,
                tool_args_match_overrides,
            ):
                return False
    return True


def trajectory_strict_match(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    tool_call_args_exact_match: bool = True,
    **kwargs: Any,
):
    """
    DEPRECATED: Use create_trajectory_match_evaluator() instead:
    ```python
    from openevals.trajectory.match import create_trajectory_match_evaluator
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    evaluator(outputs=outputs, reference_outputs=reference_outputs)
    ```
    """
    warn(
        "trajectory_strict_match() is deprecated. Use create_trajectory_match_evaluator() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    def wrapper(**kwargs: Any):
        return _scorer(
            tool_args_match_mode="exact" if tool_call_args_exact_match else "ignore",
            **kwargs,
        )

    return _run_evaluator(
        run_name="trajectory_strict_match",
        scorer=wrapper,
        feedback_key="trajectory_strict_match",
        outputs=outputs,
        reference_outputs=reference_outputs,
        **kwargs,
    )


async def trajectory_strict_match_async(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    tool_call_args_exact_match: bool = True,
    **kwargs: Any,
):
    """
    DEPRECATED: Use create_async_trajectory_match_evaluator() instead:
    ```python
    from openevals.trajectory.match import create_async_trajectory_match_evaluator
    evaluator = create_async_trajectory_match_evaluator(trajectory_match_mode="strict")
    await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    ```
    """
    warn(
        "trajectory_strict_match_async() is deprecated. Use create_async_trajectory_match_evaluator() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    def wrapper(**kwargs: Any):
        return _scorer(
            tool_args_match_mode="exact" if tool_call_args_exact_match else "ignore",
            **kwargs,
        )

    return await _arun_evaluator(
        run_name="trajectory_strict_match",
        scorer=wrapper,
        feedback_key="trajectory_strict_match",
        outputs=outputs,
        reference_outputs=reference_outputs,
        **kwargs,
    )
