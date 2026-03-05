from typing import Literal, Optional, Union

from openevals.trajectory.strict import _scorer as trajectory_strict_scorer
from openevals.trajectory.unordered import _scorer as trajectory_unordered_scorer
from openevals.trajectory.subset import _scorer as trajectory_subset_scorer
from openevals.trajectory.superset import _scorer as trajectory_superset_scorer
from openevals.types import (
    ChatCompletionMessage,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
    ToolArgsMatchMode,
    ToolArgsMatchOverrides,
)
from openevals.utils import _run_evaluator, _arun_evaluator

from openevals.trajectory.utils import _normalize_to_openai_messages_list

from langchain_core.messages import BaseMessage


TrajectoryMatchMode = Literal["strict", "unordered", "subset", "superset"]


def create_trajectory_match_evaluator(
    *,
    trajectory_match_mode: TrajectoryMatchMode = "strict",
    tool_args_match_mode: ToolArgsMatchMode = "exact",
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides] = None,
) -> SimpleEvaluator:
    """Creates an evaluator that compares trajectories between model outputs and reference outputs.

    Args:
        trajectory_match_mode (TrajectoryMatchMode): The mode for matching trajectories:
            - "strict": Requires exact match in order and content
            - "unordered": Allows matching in any order
            - "subset": Accepts if output trajectory is a subset of reference
            - "superset": Accepts if output trajectory is a superset of reference
        tool_args_match_mode (ToolArgsMatchMode): Mode for matching tool arguments ("exact" by default, can be "ignore")
        tool_args_match_overrides (Optional[ToolArgsMatchOverrides]): Dict containing custom overrides for
            tool argument matching. Each key should be a tool name, and each value should be either a
            match mode or a matcher. Matchers should be a Callable that takes two sets of tool call args
            and returns whether they are equal.

    Returns:
        SimpleEvaluator: A function that evaluates trajectory matches between outputs and references

    Example:
    ```python
    def matcher(output_tool_call_args: dict, reference_tool_call_args: dict) -> bool:
        output_args = output_tool_call_args.get("query", "").lower()
        reference_args = reference_tool_call_args.get("query", "").lower()
        return output_args == reference_args

    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="strict",
        tool_args_match_mode="exact",
        tool_args_match_overrides={
            "my_tool_name": matcher,
        },
    )
    result = evaluator(
        outputs=...,
        reference_outputs=...,
    )
    ```
    """
    if trajectory_match_mode == "strict":
        scorer = trajectory_strict_scorer
    elif trajectory_match_mode == "unordered":
        scorer = trajectory_unordered_scorer
    elif trajectory_match_mode == "subset":
        scorer = trajectory_subset_scorer
    elif trajectory_match_mode == "superset":
        scorer = trajectory_superset_scorer
    else:
        raise ValueError(
            f"Invalid trajectory match type: `{trajectory_match_mode}`. Must be one of `strict`, `unordered`, `subset`, or `superset`."
        )

    if tool_args_match_mode not in ["exact", "ignore", "subset", "superset"]:
        raise ValueError(
            f"Invalid tool args match mode: `{tool_args_match_mode}`. Must be either `exact`, `ignore`, `subset`, or `superset`."
        )

    def _wrapped_evaluator(
        *,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        **kwargs,
    ):
        outputs = _normalize_to_openai_messages_list(outputs)
        reference_outputs = _normalize_to_openai_messages_list(reference_outputs)
        return _run_evaluator(
            run_name=f"trajectory_{trajectory_match_mode}_match",
            scorer=scorer,
            feedback_key=f"trajectory_{trajectory_match_mode}_match",
            outputs=outputs,
            reference_outputs=reference_outputs,
            tool_args_match_mode=tool_args_match_mode,
            tool_args_match_overrides=tool_args_match_overrides,
            **kwargs,
        )

    return _wrapped_evaluator


def create_async_trajectory_match_evaluator(
    *,
    trajectory_match_mode: TrajectoryMatchMode = "strict",
    tool_args_match_mode: ToolArgsMatchMode = "exact",
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides] = None,
) -> SimpleAsyncEvaluator:
    """Creates an async evaluator that compares trajectories between model outputs and reference outputs.

    Args:
        trajectory_match_mode (TrajectoryMatchMode): The mode for matching trajectories:
            - "strict": Requires exact match in order and content
            - "unordered": Allows matching in any order
            - "subset": Accepts if output trajectory is a subset of reference
            - "superset": Accepts if output trajectory is a superset of reference
        tool_args_match_mode (ToolArgsMatchMode): Mode for matching tool arguments ("exact" by default, can be "ignore")
        tool_args_match_overrides (Optional[ToolArgsMatchOverrides]): Dict containing custom overrides for
            tool argument matching. Each key should be a tool name, and each value should be either a
            match mode or a matcher. Matchers should be a Callable that takes two sets of tool call args
            and returns whether they are equal.

    Returns:
        SimpleAsyncEvaluator: An async function that evaluates trajectory matches between outputs and references

    Example:
    ```python
    def matcher(output_tool_call_args: dict, reference_tool_call_args: dict) -> bool:
        output_args = output_tool_call_args.get("query", "").lower()
        reference_args = reference_tool_call_args.get("query", "").lower()
        return output_args == reference_args

    evaluator = create_async_trajectory_match_evaluator(
        trajectory_match_mode="strict",
        tool_args_match_mode="exact",
        tool_args_match_overrides={
            "my_tool_name": matcher,
        },
    )
    result = await evaluator(
        outputs=...,
        reference_outputs=...,
    )
    ```
    """
    if trajectory_match_mode == "strict":
        scorer = trajectory_strict_scorer
    elif trajectory_match_mode == "unordered":
        scorer = trajectory_unordered_scorer
    elif trajectory_match_mode == "subset":
        scorer = trajectory_subset_scorer
    elif trajectory_match_mode == "superset":
        scorer = trajectory_superset_scorer
    else:
        raise ValueError(
            f"Invalid trajectory match type: `{trajectory_match_mode}`. Must be one of `strict`, `unordered`, `subset`, or `superset`."
        )

    if tool_args_match_mode not in ["exact", "ignore", "subset", "superset"]:
        raise ValueError(
            f"Invalid tool args match mode: `{tool_args_match_mode}`. Must be either `exact`, `ignore`, `subset`, or `superset`."
        )

    async def _wrapped_evaluator(
        *,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        reference_outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        **kwargs,
    ):
        outputs = _normalize_to_openai_messages_list(outputs)
        reference_outputs = _normalize_to_openai_messages_list(reference_outputs)
        return await _arun_evaluator(
            run_name=f"trajectory_{trajectory_match_mode}_match",
            scorer=scorer,
            feedback_key=f"trajectory_{trajectory_match_mode}_match",
            outputs=outputs,
            reference_outputs=reference_outputs,
            tool_args_match_mode=tool_args_match_mode,
            tool_args_match_overrides=tool_args_match_overrides,
            **kwargs,
        )

    return _wrapped_evaluator
