import uuid
import asyncio
from typing import Any, Awaitable, Callable, Literal, Optional, Union
from openevals.types import (
    SimpleEvaluator,
    SimpleAsyncEvaluator,
    Messages,
    ChatCompletionMessage,
)
from openevals.types import (
    MultiturnSimulatorResult,
)
from openevals.utils import (
    _convert_to_openai_message,
    _normalize_to_openai_messages_list,
)
from langsmith import traceable

from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage


def _wrap(app: Callable[..., Any], run_name: str, thread_id: str) -> Runnable:
    @traceable(name=run_name)
    def _wrap_app(inputs: ChatCompletionMessage, **kwargs):
        return app(inputs, thread_id=thread_id, **kwargs)

    return _wrap_app


def _awrap(
    app: Callable[..., Awaitable[Any]], run_name: str, thread_id: str
) -> Runnable:
    @traceable(name=run_name)
    async def _wrap_app(inputs: ChatCompletionMessage, **kwargs):
        if asyncio.iscoroutinefunction(app):
            return await app(inputs, thread_id=thread_id, **kwargs)
        return app(inputs, thread_id=thread_id, **kwargs)

    return _wrap_app


def _coerce_and_assign_id_to_message(message: Union[dict, BaseMessage]) -> dict:
    message = _convert_to_openai_message(message)
    if message.get("id") is None:
        return {**message, "id": str(uuid.uuid4())}
    return message


def _trajectory_reducer(
    current_trajectory: Optional[dict],
    new_update: ChatCompletionMessage,
    *,
    update_source: Literal["app", "user"],
    turn_counter: Optional[int] = None,
) -> dict:
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
            m for m in [_coerce_and_assign_id_to_message(msg) for msg in left]
        ]
        coerced_right: list[ChatCompletionMessage] = [
            m for m in [_coerce_and_assign_id_to_message(msg) for msg in right]
        ]

        # merge
        merged = coerced_left.copy()
        merged_by_id = {m.get("id"): i for i, m in enumerate(merged)}
        for m in coerced_right:
            if merged_by_id.get(m.get("id")) is None:
                merged_by_id[m.get("id")] = len(merged)
                merged.append(m)
        return merged  # type: ignore

    if current_trajectory is None:
        current_trajectory = {"trajectory": []}

    try:
        coerced_new_update = _normalize_to_openai_messages_list(new_update)
    except ValueError as e:
        raise ValueError(
            f"Received unexpected trajectory update from '{update_source}': {str(new_update)}. Expected a message, list of messages, or dictionary with a 'messages' key containing messages."
        )
    return {
        "trajectory": _combine_messages(
            current_trajectory["trajectory"],
            coerced_new_update,
        ),
        "turn_counter": turn_counter,
    }


def _create_static_simulated_user(
    static_responses: list[str | Messages],
):
    def _return_next_message(
        trajectory: list[ChatCompletionMessage], *, thread_id: str, turn_counter: int
    ):
        turns = turn_counter
        if turns is None or not isinstance(turns, int):
            raise ValueError(
                "Internal error: Turn counter must be an integer in the trajectory."
            )
        # First conversation turn is satisfied by the initial input
        if turns >= len(static_responses):
            raise ValueError(
                "Number of conversation turns is greater than the number of static user responses. Please reduce the number of turns or provide more responses."
            )
        next_response = static_responses[turns]
        if isinstance(next_response, str):
            next_response = {
                "role": "user",
                "content": next_response,
                "id": str(uuid.uuid4()),
            }
        return next_response

    return _return_next_message


def run_multiturn_simulation(
    *,
    app: Callable[[ChatCompletionMessage], ChatCompletionMessage],
    user: Union[
        Runnable[ChatCompletionMessage, ChatCompletionMessage],
        Callable[[ChatCompletionMessage], ChatCompletionMessage],
        list[Union[str, Messages]],
    ],
    max_turns: Optional[int] = None,
    trajectory_evaluators: Optional[list[SimpleEvaluator]] = None,
    stopping_condition: Optional[Callable[[dict], bool]] = None,
    reference_outputs: Optional[Any] = None,
    thread_id: Optional[str] = None,
) -> MultiturnSimulatorResult:
    """Runs a multi-turn simulation between an application and a simulated user.

    This function simulates a conversation between an app and either a dynamic
    user simulator or a list of static user responses. The simulator supports
    evaluation of conversation trajectories and customizable stopping conditions.

    Conversation trajectories are represented as a dict containing a key named "messages" whose
    value is a list of message objects with "role" and "content" keys. The "app" and "user"
    params both receive this trajectory as an input, and should return a
    trajectory update dict with a new message or new messages under the "messages" key. The simulator
    will dedupe these messages by id and merge them into the complete trajectory.

    Additional fields are also permitted as part of the trajectory dict, which allows you to pass
    additional information between the app and user if needed.

    Once "max_turns" is reached or a provided stopping condition is met, the final trajectory
    will be passed to provided trajectory evaluators, which will receive the final trajectory
    as an "outputs" kwarg.

    Args:
        app: Your application. Can be either a LangChain Runnable or a
            callable that takes the current conversation trajectory dict and returns
            a trajectory update dict with new messages under the "messages" key (and optionally other fields).
        user: The simulated user. Can be:
            - A LangChain Runnable or a callable that takes the current conversation trajectory
              and returns a trajectory update dict with new messages under the "messages" key (and optionally other fields).
            - A list of strings or Messages representing static user responses
        max_turns: Maximum number of conversation turns to simulate
        trajectory_evaluators: Optional list of evaluator functions that assess the conversation
            trajectory. Each evaluator will receive the final trajectory of the conversation as
            a kwarg named "outputs" and a kwarg named "reference_outputs" if provided.
        stopping_condition: Optional callable that determines if the simulation should end early.
            Takes the current trajectory and turn counter as input and returns a boolean.
        reference_outputs: Optional reference outputs for evaluation
        thread_id: Thread ID for the simulation

    Returns:
        A MultiturnSimulatorResult containing:
            - evaluator_results: List of results from trajectory evaluators
            - trajectory: The complete conversation trajectory

    Example:
        ```python
        from openevals.simulators import run_multiturn_simulation

        # Run a simulation directly
        result = run_multiturn_simulation(
            app=my_chat_app,
            user=["Hello!", "How are you?", "Goodbye"],
            max_turns=3,
            trajectory_evaluators=[my_evaluator],
            thread_id="simulation-1"
        )
        ```
    """
    simulator = _create_multiturn_simulator(
        app=app,
        user=user,
        max_turns=max_turns,
        trajectory_evaluators=trajectory_evaluators,
        stopping_condition=stopping_condition,
    )

    return simulator(reference_outputs=reference_outputs, thread_id=thread_id)


def _create_multiturn_simulator(
    *,
    app: Callable[[ChatCompletionMessage], ChatCompletionMessage],
    user: Union[
        Runnable[ChatCompletionMessage, ChatCompletionMessage],
        Callable[[ChatCompletionMessage], ChatCompletionMessage],
        list[Union[str, Messages]],
    ],
    max_turns: Optional[int] = None,
    trajectory_evaluators: Optional[list[SimpleEvaluator]] = None,
    stopping_condition: Optional[Callable[[dict], bool]] = None,
) -> Callable[..., MultiturnSimulatorResult]:
    if max_turns is None and stopping_condition is None:
        raise ValueError(
            "At least one of max_turns or stopping_condition must be provided."
        )

    @traceable(name="multiturn_simulator")
    def _run_simulator(
        *,
        reference_outputs: Optional[Any] = None,
        thread_id: str,
    ):
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        turn_counter = 0
        current_reduced_trajectory = {
            "trajectory": [],
            "turn_counter": 0,
        }
        wrapped_app = _wrap(app, "app", thread_id)
        if isinstance(user, list):
            static_responses = user
            simulated_user = _create_static_simulated_user(static_responses)
        else:
            simulated_user = user  # type: ignore
        wrapped_simulated_user = _wrap(simulated_user, "simulated_user", thread_id)

        while True:
            if max_turns is not None and turn_counter >= max_turns:
                break
            raw_inputs = wrapped_simulated_user(
                current_reduced_trajectory["trajectory"], turn_counter=turn_counter
            )
            current_inputs = _coerce_and_assign_id_to_message(raw_inputs)
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory,
                current_inputs,
                update_source="user",
                turn_counter=turn_counter,
            )
            raw_outputs = wrapped_app(current_inputs)
            current_outputs = _coerce_and_assign_id_to_message(raw_outputs)
            turn_counter += 1
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory,
                current_outputs,
                update_source="app",
                turn_counter=turn_counter,
            )
            if stopping_condition and stopping_condition(
                current_reduced_trajectory["trajectory"], turn_counter=turn_counter
            ):
                break
        results = []
        del current_reduced_trajectory["turn_counter"]
        for trajectory_evaluator in trajectory_evaluators or []:
            try:
                trajectory_eval_result = trajectory_evaluator(
                    outputs=current_reduced_trajectory["trajectory"],
                    reference_outputs=reference_outputs,
                )
                if isinstance(trajectory_eval_result, list):
                    results.extend(trajectory_eval_result)
                else:
                    results.append(trajectory_eval_result)
            except Exception as e:
                print(f"Error in trajectory evaluator {trajectory_evaluator}: {e}")
        return MultiturnSimulatorResult(
            trajectory=current_reduced_trajectory["trajectory"],
            evaluator_results=results,
        )

    return _run_simulator


async def run_multiturn_simulation_async(
    *,
    app: Callable[[ChatCompletionMessage], Awaitable[ChatCompletionMessage]],
    user: Union[
        Runnable[ChatCompletionMessage, ChatCompletionMessage],
        Callable[[ChatCompletionMessage], Awaitable[ChatCompletionMessage]],
        list[Union[str, Messages]],
    ],
    max_turns: Optional[int] = None,
    trajectory_evaluators: Optional[list[SimpleAsyncEvaluator]] = None,
    stopping_condition: Optional[Callable[[dict], bool]] = None,
    reference_outputs: Optional[Any] = None,
    thread_id: Optional[str] = None,
) -> MultiturnSimulatorResult:
    """Runs an async multi-turn simulation between an application and a simulated user.

    This function simulates a conversation between an app and either a dynamic
    user simulator or a list of static user responses. The simulator supports
    evaluation of conversation trajectories and customizable stopping conditions.

    Conversation trajectories are represented as a dict containing a key named "messages" whose
    value is a list of message objects with "role" and "content" keys. The "app" and "user"
    params both receive this trajectory as an input, and should return a
    trajectory update dict with a new message or new messages under the "messages" key. The simulator
    will dedupe these messages by id and merge them into the complete trajectory.

    Additional fields are also permitted as part of the trajectory dict, which allows you to pass
    additional information between the app and user if needed.

    Once "max_turns" is reached or a provided stopping condition is met, the final trajectory
    will be passed to provided trajectory evaluators, which will receive the final trajectory
    as an "outputs" kwarg.

    Args:
        app: Your application. Can be either a LangChain Runnable or a
            callable that takes the current conversation trajectory dict and returns
            a trajectory update dict with new messages under the "messages" key (and optionally other fields).
        user: The simulated user. Can be:
            - A LangChain Runnable or a callable that takes the current conversation trajectory
              and returns a trajectory update dict with new messages under the "messages" key (and optionally other fields).
            - A list of strings or Messages representing static user responses
        max_turns: Maximum number of conversation turns to simulate
        trajectory_evaluators: Optional list of evaluator functions that assess the conversation
            trajectory. Each evaluator will receive the final trajectory of the conversation as
            a kwarg named "outputs" and a kwarg named "reference_outputs" if provided.
        stopping_condition: Optional callable that determines if the simulation should end early.
            Takes the current trajectory and turn counter as input and returns a boolean.
        reference_outputs: Optional reference outputs for evaluation
        thread_id: Thread ID for the simulation

    Returns:
        A MultiturnSimulatorResult containing:
            - evaluator_results: List of results from trajectory evaluators
            - trajectory: The complete conversation trajectory

    Example:
        ```python
        from openevals.simulators import run_multiturn_simulation_async

        # Run a simulation directly
        result = await run_multiturn_simulation_async(
            app=my_chat_app,
            user=["Hello!", "How are you?", "Goodbye"],
            max_turns=3,
            trajectory_evaluators=[my_evaluator],
            thread_id="simulation-1"
        )
        ```
    """
    simulator = _create_async_multiturn_simulator(
        app=app,
        user=user,
        max_turns=max_turns,
        trajectory_evaluators=trajectory_evaluators,
        stopping_condition=stopping_condition,
    )

    return await simulator(reference_outputs=reference_outputs, thread_id=thread_id)


def _create_async_multiturn_simulator(
    *,
    app: Callable[[ChatCompletionMessage], Awaitable[ChatCompletionMessage]],
    user: Union[
        Runnable[ChatCompletionMessage, ChatCompletionMessage],
        Callable[[ChatCompletionMessage], Awaitable[ChatCompletionMessage]],
        list[Union[str, Messages]],
    ],
    max_turns: Optional[int] = None,
    trajectory_evaluators: Optional[list[SimpleAsyncEvaluator]] = None,
    stopping_condition: Optional[Callable[[dict], bool]] = None,
) -> Callable[..., Awaitable[MultiturnSimulatorResult]]:
    if max_turns is None and stopping_condition is None:
        raise ValueError(
            "At least one of max_turns or stopping_condition must be provided."
        )

    @traceable(name="multiturn_simulator")
    async def _run_simulator(
        *,
        reference_outputs: Optional[Any] = None,
        thread_id: Optional[str] = None,
    ):
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        turn_counter = 0
        current_reduced_trajectory = {
            "trajectory": [],
            "turn_counter": 0,
        }
        wrapped_app = _awrap(app, "app", thread_id)
        if isinstance(user, list):
            static_responses = user
            simulated_user = _create_static_simulated_user(static_responses)
        else:
            simulated_user = user  # type: ignore
        wrapped_simulated_user = _awrap(simulated_user, "simulated_user", thread_id)

        while True:
            if max_turns is not None and turn_counter >= max_turns:
                break
            raw_inputs = await wrapped_simulated_user(
                current_reduced_trajectory["trajectory"], turn_counter=turn_counter
            )
            current_inputs = _coerce_and_assign_id_to_message(raw_inputs)
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory,
                current_inputs,
                update_source="user",
                turn_counter=turn_counter,
            )
            raw_outputs = await wrapped_app(current_inputs)
            current_outputs = _coerce_and_assign_id_to_message(raw_outputs)
            turn_counter += 1
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory,
                current_outputs,
                update_source="app",
                turn_counter=turn_counter,
            )
            if stopping_condition and await stopping_condition(
                current_reduced_trajectory["trajectory"], turn_counter=turn_counter
            ):
                break
        results = []
        del current_reduced_trajectory["turn_counter"]
        for trajectory_evaluator in trajectory_evaluators or []:
            try:
                trajectory_eval_result = await trajectory_evaluator(
                    outputs=current_reduced_trajectory["trajectory"],
                    reference_outputs=reference_outputs,
                )
                if isinstance(trajectory_eval_result, list):
                    results.extend(trajectory_eval_result)
                else:
                    results.append(trajectory_eval_result)
            except Exception as e:
                print(f"Error in trajectory evaluator {trajectory_evaluator}: {e}")
        return MultiturnSimulatorResult(
            trajectory=current_reduced_trajectory["trajectory"],
            evaluator_results=results,
        )

    return _run_simulator
