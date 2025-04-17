import uuid
from typing import Any, Callable, Optional, Union
from openevals.types import (
    SimpleEvaluator,
    Messages,
    MessagesDict,
    ChatCompletionMessage,
)
from openevals.utils import _convert_to_openai_message
from langsmith import traceable

from langchain_core.runnables import RunnableLambda, Runnable


def _wrap(app: Runnable | Callable[..., Any], run_name: str) -> Runnable:
    if isinstance(app, Runnable):
        return app
    else:
        return RunnableLambda(app).with_config({"run_name": run_name})


def _is_internal_message(message: ChatCompletionMessage) -> bool:
    return message.get("role") != "user" and (
        message.get("role") != "assistant" or message.get("tool_calls")
    )


def _trajectory_reducer(
    current_trajectory: Optional[Union[list[Messages], MessagesDict]],
    new_update: Union[list[Messages], MessagesDict],
) -> Union[list[Messages], MessagesDict]:
    def _combine_messages(
        left: list[Messages],
        right: list[Messages],
    ) -> list[Messages]:
        remove_all_idx = None
        # coerce to list
        if not isinstance(left, list):
            left = [left]  # type: ignore[assignment]
        if not isinstance(right, list):
            right = [right]  # type: ignore[assignment]
        # coerce to message
        coerced_left: list[ChatCompletionMessage] = [
            m
            for m in [_convert_to_openai_message(msg) for msg in left]  # type: ignore
            if not _is_internal_message(m)
        ]
        coerced_right: list[ChatCompletionMessage] = [
            m
            for m in [_convert_to_openai_message(msg) for msg in right]  # type: ignore
            if not _is_internal_message(m)
        ]
        # assign missing ids
        for m in coerced_left:
            if m.get("id") is None:
                m["id"] = str(uuid.uuid4())
        for m in coerced_right:
            if m.get("id") is None:
                m["id"] = str(uuid.uuid4())

        if remove_all_idx is not None:
            return coerced_right[remove_all_idx + 1 :]  # type: ignore

        # merge
        merged = coerced_left.copy()
        merged_by_id = {m.get("id"): i for i, m in enumerate(merged)}
        for m in coerced_right:
            if merged_by_id.get(m.get("id")) is None:
                merged_by_id[m.get("id")] = len(merged)
                merged.append(m)
        return merged  # type: ignore

    if current_trajectory is None:
        if isinstance(new_update, list):
            current_trajectory = []
        elif isinstance(new_update, dict):
            if "messages" in new_update and isinstance(new_update["messages"], list):
                current_trajectory = {"messages": []}
    if isinstance(current_trajectory, list) and isinstance(new_update, list):
        return _combine_messages(current_trajectory, new_update)
    elif isinstance(current_trajectory, dict) and isinstance(new_update, dict):
        if (
            "messages" in current_trajectory
            and isinstance(current_trajectory["messages"], list)
            and "messages" in new_update
            and isinstance(new_update["messages"], list)
        ):
            return {
                **current_trajectory,
                **new_update,
                "messages": _combine_messages(
                    current_trajectory["messages"],
                    new_update["messages"],
                ),
            }
        else:
            raise ValueError(f"Unexpected trajectory format: {type(new_update)}")
    else:
        raise ValueError(f"Unexpected trajectory format: {type(new_update)}")


def create_multiturn_simulator(
    *,
    app: Runnable | Callable[..., Any],
    user: Runnable | Callable[..., Any] | list[Any],
    trajectory_evaluators: list[SimpleEvaluator],
    max_turns: int = 5,
    # stopping_condition: Optional[Callable[[list[Any]], bool]] = None,
) -> SimpleEvaluator:
    if not trajectory_evaluators:
        raise ValueError("You must pass at least one trajectory evaluator.")

    @traceable(name="multiturn_simulator")
    def _run_simulator(
        *,
        inputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ):
        wrapped_app = _wrap(app, "app")
        wrapped_simulated_user = _wrap(user, "simulated_user")
        turn_counter = 0
        raw_trajectory = []
        current_reduced_trajectory = None
        while turn_counter < max_turns:
            current_inputs = (
                inputs
                if turn_counter == 0
                else wrapped_simulated_user.invoke(current_reduced_trajectory)
            )
            raw_trajectory.append(current_inputs)
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory, current_inputs
            )
            current_outputs = wrapped_app.invoke(current_reduced_trajectory)
            raw_trajectory.append(current_outputs)
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory, current_outputs
            )
            turn_counter += 1
        results = []
        for trajectory_evaluator in trajectory_evaluators:
            trajectory_eval_result = trajectory_evaluator(
                outputs=current_reduced_trajectory,
                reference_outputs=reference_outputs,
            )
            results.append(trajectory_eval_result)
        return {
            "results": results,
            "trajectory": current_reduced_trajectory,
        }

    return _run_simulator
