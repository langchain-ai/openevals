import uuid
from typing import Any, Callable, Optional, Union, cast
from openevals.types import (
    SimpleEvaluator,
    ChatCompletionMessage,
)
from openevals.utils import _run_evaluator, _convert_to_openai_message

from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)


def _wrap(app: Runnable | Callable[..., Any], run_name: str) -> Runnable:
    if isinstance(app, Runnable):
        return app
    else:
        return RunnableLambda(app).with_config({"run_name": run_name})


REMOVE_ALL_MESSAGES = "__remove_all__"


def _default_trajectory_reducer(current_trajectory: Any, new_update: Any) -> Any:
    def _add_messages(
        left: list[Union[BaseMessage, ChatCompletionMessage]],
        right: list[Union[BaseMessage, ChatCompletionMessage]],
    ) -> list[Union[BaseMessage, ChatCompletionMessage]]:
        remove_all_idx = None
        # coerce to list
        if not isinstance(left, list):
            left = [left]  # type: ignore[assignment]
        if not isinstance(right, list):
            right = [right]  # type: ignore[assignment]
        # coerce to message
        coerced_left: list[BaseMessage] = [
            message_chunk_to_message(cast(BaseMessageChunk, m))
            for m in convert_to_messages(left)  # type: ignore
        ]
        coerced_right: list[BaseMessage] = [
            message_chunk_to_message(cast(BaseMessageChunk, m))
            for m in convert_to_messages(right)  # type: ignore
        ]
        # assign missing ids
        for m in coerced_left:
            if m.id is None:
                m.id = str(uuid.uuid4())
        for idx, m in enumerate(coerced_right):
            if m.id is None:
                m.id = str(uuid.uuid4())
            if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
                remove_all_idx = idx

        if remove_all_idx is not None:
            return coerced_right[remove_all_idx + 1 :]  # type: ignore

        # merge
        merged = coerced_left.copy()
        merged_by_id = {m.id: i for i, m in enumerate(merged)}
        ids_to_remove = set()
        for m in coerced_right:
            if (existing_idx := merged_by_id.get(m.id)) is not None:
                if isinstance(m, RemoveMessage):
                    ids_to_remove.add(m.id)
                else:
                    ids_to_remove.discard(m.id)
                    merged[existing_idx] = m
            else:
                if isinstance(m, RemoveMessage):
                    raise ValueError(
                        f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                    )

                merged_by_id[m.id] = len(merged)
                merged.append(m)
        merged = [
            _convert_to_openai_message(m)
            for m in merged
            if m.id not in ids_to_remove  # type: ignore
        ]
        return merged  # type: ignore

    if current_trajectory is None:
        if isinstance(new_update, list):
            current_trajectory = []
        elif isinstance(new_update, dict):
            if "messages" in new_update and isinstance(new_update["messages"], list):
                current_trajectory = {"messages": []}
    if isinstance(current_trajectory, list) and isinstance(new_update, list):
        return _add_messages(current_trajectory, new_update)
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
                "messages": _add_messages(
                    current_trajectory["messages"],
                    new_update["messages"],
                ),
            }
        else:
            raise ValueError(f"Unexpected trajectory format: {type(new_update)}")
    else:
        raise ValueError(f"Unexpected trajectory format: {type(new_update)}")


def create_multiturn_evaluator(
    *,
    app: Runnable | Callable[..., Any],
    simulator: Runnable | Callable[..., Any] | list[Any],
    trajectory_evaluators: Optional[list[SimpleEvaluator]] = None,
    single_turn_evaluators: Optional[list[SimpleEvaluator]] = None,
    trajectory_reducer: Optional[Callable[[Any, Any], Any]] = None,
    max_turns: int = 5,
    feedback_key: Optional[str] = None,
    # stopping_condition: Optional[Callable[[list[Any]], bool]] = None,
) -> SimpleEvaluator:
    if feedback_key is None:
        feedback_key = f"pass<={max_turns}"

    if not trajectory_evaluators and not single_turn_evaluators:
        raise ValueError(
            "At least one trajectory or single turn evaluator must be provided."
        )

    if not trajectory_reducer:
        trajectory_reducer = _default_trajectory_reducer  # type: ignore

    def _scorer(
        *,
        inputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ):
        single_turn_eval_results = []
        wrapped_app = _wrap(app, "app")
        wrapped_simulator = _wrap(simulator, "simulator")
        turn_counter = 0
        raw_trajectory = []
        current_reduced_trajectory = None
        remaining_trajectory_evaluators = list(trajectory_evaluators or [])
        while turn_counter < max_turns:
            current_inputs = (
                inputs
                if turn_counter == 0
                else wrapped_simulator.invoke(current_reduced_trajectory)
            )
            raw_trajectory.append(current_inputs)
            current_reduced_trajectory = trajectory_reducer(
                current_reduced_trajectory, current_inputs
            )
            current_outputs = wrapped_app.invoke(current_reduced_trajectory)
            raw_trajectory.append(current_outputs)
            current_reduced_trajectory = trajectory_reducer(
                current_reduced_trajectory, current_outputs
            )
            turn_counter += 1
            for turn_evaluator in single_turn_evaluators or []:
                turn_eval_result = turn_evaluator(
                    inputs=raw_trajectory[-2], outputs=raw_trajectory[-1]
                )
                if turn_eval_result.get("metadata") is None:
                    turn_eval_result["metadata"] = {}
                turn_eval_result["metadata"]["turn"] = turn_counter
                single_turn_eval_results.append(turn_eval_result)
                # TODO: Make threshold customizable
                if turn_eval_result["score"] < 0.5 and not trajectory_evaluators:
                    return (
                        False,
                        "Single turn evaluator failed with no passed trajectory evaluators.",
                        {"turn": turn_counter},
                    )
            for trajectory_evaluator in remaining_trajectory_evaluators:
                trajectory_eval_result = trajectory_evaluator(
                    outputs=current_reduced_trajectory,
                    reference_outputs=reference_outputs,
                )
                if trajectory_eval_result.get("metadata") is None:
                    trajectory_eval_result["metadata"] = {}
                trajectory_eval_result["metadata"]["turn"] = turn_counter
                # TODO: Make threshold customizable
                if trajectory_eval_result["score"] >= 0.5:
                    remaining_trajectory_evaluators.remove(trajectory_evaluator)
            if trajectory_evaluators and len(remaining_trajectory_evaluators) == 0:
                break
        if len(remaining_trajectory_evaluators) == 0:
            return (
                True,
                f"All trajectory evaluators passed in {turn_counter} turns.",
                {"turn": turn_counter},
            )
        else:
            return (
                False,
                f"Some trajectory evaluators still failing after {turn_counter} turns.",
                {"turn": turn_counter},
            )

    def _wrapped_evaluator(
        *,
        inputs: Any,
        **kwargs,
    ):
        res = _run_evaluator(
            run_name="multiturn_evaluator",
            scorer=_scorer,
            feedback_key=feedback_key,
            inputs=inputs,
            **kwargs,
        )
        return res

    return _wrapped_evaluator
