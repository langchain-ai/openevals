import uuid
from typing import Any, Callable, Literal, Optional, Union, Protocol, runtime_checkable
from openevals.types import (
    SimpleEvaluator,
    Messages,
    MessagesDict,
    MessagesDictUpdate,
    ChatCompletionMessage,
)
from openevals.utils import _convert_to_openai_message
from langsmith import traceable

from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig

from openevals.simulators.multiturn.types import MultiturnSimulatorResult


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
    current_trajectory: Optional[MessagesDict],
    new_update: MessagesDictUpdate,
    *,
    update_source: Literal["app", "user"],
) -> MessagesDict:
    def _combine_messages(
        left: list[Messages] | Messages,
        right: list[Messages] | Messages,
    ) -> list[Messages]:
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

        # merge
        merged = coerced_left.copy()
        merged_by_id = {m.get("id"): i for i, m in enumerate(merged)}
        for m in coerced_right:
            if merged_by_id.get(m.get("id")) is None:
                merged_by_id[m.get("id")] = len(merged)
                merged.append(m)
        return merged  # type: ignore

    if current_trajectory is None:
        current_trajectory = {"messages": []}
    if isinstance(new_update, dict) and "messages" in new_update:
        return {
            **current_trajectory,
            **new_update,
            "messages": _combine_messages(
                current_trajectory["messages"],
                new_update["messages"],
            ),
        }
    else:
        raise ValueError(
            f"Received unexpected trajectory update from {update_source}: {str(new_update)}. Expected a dictionary with a 'messages' key."
        )


@runtime_checkable
class StoppingCondition(Protocol):
    def __call__(
        self,
        current_trajectory: MessagesDict,
        *,
        turn_counter: int,
        **kwargs,
    ) -> bool: ...


def create_multiturn_simulator(
    *,
    app: Runnable[MessagesDict, MessagesDictUpdate]
    | Callable[[MessagesDict], MessagesDictUpdate],
    user: Runnable[MessagesDict, MessagesDictUpdate]
    | Callable[[MessagesDict], MessagesDictUpdate]
    | list[str | Messages],
    trajectory_evaluators: Optional[list[SimpleEvaluator]] = None,
    max_turns: int = 5,
    stopping_condition: Optional[StoppingCondition] = None,
    runnable_config: Optional[RunnableConfig] = None,
) -> SimpleEvaluator:
    @traceable(name="multiturn_simulator")
    def _run_simulator(
        *,
        inputs: Any,
        reference_outputs: Optional[Any] = None,
        **kwargs,
    ):
        turn_counter = 0
        current_reduced_trajectory = None
        wrapped_app = _wrap(app, "app")
        if isinstance(user, list):
            static_responses = user
            call_counter = 0

            def _return_next_message(
                trajectory: Optional[MessagesDictUpdate],
            ):
                nonlocal call_counter
                if call_counter >= len(static_responses):
                    raise ValueError(
                        "Number of conversation turns is greater than the number of static user responses. Please reduce the number of turns or provide more responses."
                    )
                next_response = static_responses[call_counter]
                if isinstance(next_response, str):
                    next_response = {"role": "user", "content": next_response}
                call_counter += 1
                return {"messages": next_response}

            simulated_user = _return_next_message
        else:
            simulated_user = user
        wrapped_simulated_user = _wrap(simulated_user, "simulated_user")
        while turn_counter < max_turns:
            current_inputs = (
                inputs
                if turn_counter == 0
                else wrapped_simulated_user.invoke(
                    current_reduced_trajectory, config=runnable_config
                )
            )
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory, current_inputs, update_source="user"
            )
            current_outputs = wrapped_app.invoke(
                current_reduced_trajectory, config=runnable_config
            )
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory, current_outputs, update_source="app"
            )
            turn_counter += 1
            if stopping_condition and stopping_condition(
                current_reduced_trajectory, turn_counter=turn_counter
            ):
                break
        results = []
        for trajectory_evaluator in trajectory_evaluators:
            trajectory_eval_result = trajectory_evaluator(
                outputs=current_reduced_trajectory,
                reference_outputs=reference_outputs,
            )
            results.append(trajectory_eval_result)
        return MultiturnSimulatorResult(
            evaluator_results=results,
            trajectory=current_reduced_trajectory,
        )

    return _run_simulator
