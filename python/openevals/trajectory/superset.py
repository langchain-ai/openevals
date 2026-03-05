from __future__ import annotations
from warnings import warn

from openevals.types import (
    ChatCompletionMessage,
    ToolArgsMatchMode,
    ToolArgsMatchOverrides,
)
from openevals.trajectory.utils import (
    _is_trajectory_superset,
    _normalize_to_openai_messages_list,
)
from openevals.utils import _run_evaluator, _arun_evaluator

from typing import Any, Optional, Union, TYPE_CHECKING

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
    if outputs is None or reference_outputs is None:
        raise ValueError(
            "Trajectory superset match requires both outputs and reference_outputs"
        )
    # superset: outputs contains all tool calls from reference_outputs
    return _is_trajectory_superset(
        outputs, reference_outputs, tool_args_match_mode, tool_args_match_overrides
    )


def trajectory_superset(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    **kwargs: Any,
):
    """
    DEPRECATED: Use create_trajectory_match_evaluator() instead:
    ```python
    from openevals.trajectory.match import create_trajectory_match_evaluator
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="superset")
    evaluator(outputs=outputs, reference_outputs=reference_outputs)
    ```
    """
    warn(
        "trajectory_superset() is deprecated. Use create_trajectory_match_evaluator(trajectory_match_mode='superset') instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)

    return _run_evaluator(
        run_name="trajectory_superset",
        scorer=_scorer,
        feedback_key="trajectory_superset",
        outputs=outputs,
        reference_outputs=reference_outputs,
        tool_args_match_mode="ignore",
        **kwargs,
    )


async def trajectory_superset_async(
    *,
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    **kwargs: Any,
):
    """
    DEPRECATED: Use create_async_trajectory_match_evaluator() instead:
    ```python
    from openevals.trajectory.match import create_async_trajectory_match_evaluator
    evaluator = create_async_trajectory_match_evaluator(trajectory_match_mode="superset")
    await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    ```
    """
    warn(
        "trajectory_superset_async() is deprecated. Use create_async_trajectory_match_evaluator(trajectory_match_mode='superset') instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)

    return await _arun_evaluator(
        run_name="trajectory_superset",
        scorer=_scorer,
        feedback_key="trajectory_superset",
        outputs=outputs,
        reference_outputs=reference_outputs,
        tool_args_match_mode="ignore",
        **kwargs,
    )
