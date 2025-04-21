import uuid
from typing import Any, Awaitable, Callable, Literal, Optional, Union
from openevals.types import (
    SimpleEvaluator,
    SimpleAsyncEvaluator,
    Messages,
    ChatCompletionMessage,
)
from openevals.types import (
    MultiturnSimulatorTrajectory,
    MultiturnSimulatorTrajectoryUpdate,
    MultiturnSimulatorResult,
)
from openevals.utils import _convert_to_openai_message
from langsmith import traceable

from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig


def _wrap(app: Union[Runnable, Callable[..., Any]], run_name: str) -> Runnable:
    if isinstance(app, Runnable):
        return app
    else:
        return RunnableLambda(app).with_config({"run_name": run_name})


def _is_internal_message(message: ChatCompletionMessage) -> bool:
    return bool(
        message.get("role") != "user"
        and (message.get("role") != "assistant" or message.get("tool_calls"))
    )


def _trajectory_reducer(
    current_trajectory: Optional[MultiturnSimulatorTrajectory],
    new_update: MultiturnSimulatorTrajectoryUpdate,
    *,
    update_source: Literal["app", "user"],
    turn_counter: int,
) -> MultiturnSimulatorTrajectory:
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
            for m in [_convert_to_openai_message(msg) for msg in left]
            if not _is_internal_message(m)
        ]
        coerced_right: list[ChatCompletionMessage] = [
            m
            for m in [_convert_to_openai_message(msg) for msg in right]
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
            "turn_counter": turn_counter,
        }
    else:
        raise ValueError(
            f"Received unexpected trajectory update from {update_source}: {str(new_update)}. Expected a dictionary with a 'messages' key."
        )


def _create_static_simulated_user(
    static_responses: list[str | Messages],
):
    def _return_next_message(
        trajectory: MultiturnSimulatorTrajectory,
    ):
        turns = trajectory.get("turn_counter")
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
            next_response = {"role": "user", "content": next_response}
        return {"messages": next_response}

    return _return_next_message


def create_multiturn_simulator(
    *,
    app: Union[
        Runnable[MultiturnSimulatorTrajectory, MultiturnSimulatorTrajectoryUpdate],
        Callable[[MultiturnSimulatorTrajectory], MultiturnSimulatorTrajectoryUpdate],
    ],
    user: Union[
        Runnable[MultiturnSimulatorTrajectory, MultiturnSimulatorTrajectoryUpdate],
        Callable[[MultiturnSimulatorTrajectory], MultiturnSimulatorTrajectoryUpdate],
        list[Union[str, Messages]],
    ],
    max_turns: Optional[int] = None,
    trajectory_evaluators: Optional[list[SimpleEvaluator]] = None,
    stopping_condition: Optional[Callable[[MultiturnSimulatorTrajectory], bool]] = None,
) -> Callable[..., MultiturnSimulatorResult]:
    """Creates a simulator for multi-turn conversations between an application and a simulated user.

    This function generates a simulator that can run conversations between an app and
    either a dynamic user simulator or a list of static user responses. The simulator supports
    evaluation of conversation trajectories and customizable stopping conditions.

    Conversation trajectories are represented as a dict containing a key named "messages" whose
    value is a list of message objects with "role" and "content" keys. The "app" and "user"
    params you provide will both receive this trajectory as an input, and should return a
    trajectory update dict with a new message or new messages under the "messages" key. The simulator
    will dedupe these messages by id and merge them into the complete trajectory.

    Additional fields are also permitted as part of the trajectory dict, which allows you to pass
    additional information between the app and user if needed.

    Once "max_turns" is reached or a provided stopping condition is met, the final trajectory
    will be passed to provided trajectory evaluators, which will receive the final trajectory
    as an "outputs" kwarg.

    When calling the created simulator, you may also provide a "reference_outputs" kwarg,
    which will be passed directly through to the provided evaluators.

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

    Returns:
        A callable that runs the simulation when invoked. The callable accepts the following kwargs:
            - initial_trajectory: Initial input to start the conversation
            - reference_outputs: Optional reference outputs for evaluation
            - runnable_config: Optional config that will be passed in if using LangChain Runnable components.
            - **kwargs: Additional keyword arguments
        Returns a MultiturnSimulatorResult containing:
            - evaluator_results: List of results from trajectory evaluators
            - trajectory: The complete conversation trajectory

    Example:
        ```python
        from openevals.simulators import create_multiturn_simulator

        # Create a simulator with static user responses
        simulator = create_multiturn_simulator(
            app=my_chat_app,
            user=["Hello!", "How are you?", "Goodbye"],
            max_turns=3,
            trajectory_evaluators=[my_evaluator]
        )

        # Run the simulation
        result = simulator(initial_trajectory={"messages": [{"role": "user", "content": "Start"}]})
        ```
    """

    if max_turns is None and stopping_condition is None:
        raise ValueError(
            "At least one of max_turns or stopping_condition must be provided."
        )

    @traceable(name="multiturn_simulator")
    def _run_simulator(
        *,
        initial_trajectory: MultiturnSimulatorTrajectory,
        reference_outputs: Optional[Any] = None,
        runnable_config: Optional[RunnableConfig] = None,
        **kwargs,
    ):
        turn_counter = 0
        current_reduced_trajectory: MultiturnSimulatorTrajectory = {"messages": []}
        wrapped_app = _wrap(app, "app")
        if isinstance(user, list):
            static_responses = user
            simulated_user = _create_static_simulated_user(static_responses)
        else:
            simulated_user = user  # type: ignore
        wrapped_simulated_user = _wrap(simulated_user, "simulated_user")
        while True:
            if max_turns is not None and turn_counter >= max_turns:
                break
            current_inputs = (
                initial_trajectory
                if turn_counter == 0
                else wrapped_simulated_user.invoke(
                    current_reduced_trajectory, config=runnable_config
                )
            )
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory,
                current_inputs,
                update_source="user",
                turn_counter=turn_counter,
            )
            current_outputs = wrapped_app.invoke(
                current_reduced_trajectory, config=runnable_config
            )
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory,
                current_outputs,
                update_source="app",
                turn_counter=turn_counter,
            )
            turn_counter += 1
            if stopping_condition and stopping_condition(current_reduced_trajectory):
                break
        results = []
        del current_reduced_trajectory["turn_counter"]
        for trajectory_evaluator in trajectory_evaluators or []:
            try:
                trajectory_eval_result = trajectory_evaluator(
                    outputs=current_reduced_trajectory,
                    reference_outputs=reference_outputs,
                )
                if isinstance(trajectory_eval_result, list):
                    results.extend(trajectory_eval_result)
                else:
                    results.append(trajectory_eval_result)
            except Exception as e:
                print(f"Error in trajectory evaluator {trajectory_evaluator}: {e}")
        return MultiturnSimulatorResult(
            trajectory=current_reduced_trajectory,
            evaluator_results=results,
        )

    return _run_simulator


def create_async_multiturn_simulator(
    *,
    app: Union[
        Runnable[MultiturnSimulatorTrajectory, MultiturnSimulatorTrajectoryUpdate],
        Callable[
            [MultiturnSimulatorTrajectory],
            Awaitable[MultiturnSimulatorTrajectoryUpdate],
        ],
    ],
    user: Union[
        Runnable[MultiturnSimulatorTrajectory, MultiturnSimulatorTrajectoryUpdate],
        Callable[
            [MultiturnSimulatorTrajectory],
            Awaitable[MultiturnSimulatorTrajectoryUpdate],
        ],
        list[Union[str, Messages]],
    ],
    max_turns: Optional[int] = None,
    trajectory_evaluators: Optional[list[SimpleAsyncEvaluator]] = None,
    stopping_condition: Optional[
        Callable[[MultiturnSimulatorTrajectory], Awaitable[bool]]
    ] = None,
) -> Callable[..., Awaitable[MultiturnSimulatorResult]]:
    """Creates an async simulator for multi-turn conversations between an application and a simulated user.

    This function generates a simulator that can run conversations between an app and
    either a dynamic user simulator or a list of static user responses. The simulator supports
    evaluation of conversation trajectories and customizable stopping conditions.

    Conversation trajectories are represented as a dict containing a key named "messages" whose
    value is a list of message objects with "role" and "content" keys. The "app" and "user"
    params you provide will both receive this trajectory as an input, and should return a
    trajectory update dict with a new message or new messages under the "messages" key. The simulator
    will dedupe these messages by id and merge them into the complete trajectory.

    Additional fields are also permitted as part of the trajectory dict, which allows you to pass
    additional information between the app and user if needed.

    Once "max_turns" is reached or a provided stopping condition is met, the final trajectory
    will be passed to provided trajectory evaluators, which will receive the final trajectory
    as an "outputs" kwarg.

    When calling the created simulator, you may also provide a "reference_outputs" kwarg,
    which will be passed directly through to the provided evaluators.

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

    Returns:
        A callable that runs the simulation when invoked. The callable accepts the following kwargs:
            - initial_trajectory: Initial input to start the conversation
            - reference_outputs: Optional reference outputs for evaluation
            - runnable_config: Optional config that will be passed in if using LangChain Runnable components.
            - **kwargs: Additional keyword arguments
        Returns an awaitable value that resolves to a MultiturnSimulatorResult containing:
            - evaluator_results: List of results from trajectory evaluators
            - trajectory: The complete conversation trajectory

    Example:
        ```python
        from openevals.simulators import create_async_multiturn_simulator

        # Create a simulator with static user responses
        simulator = create_async_multiturn_simulator(
            app=my_chat_app,
            user=["Hello!", "How are you?", "Goodbye"],
            max_turns=3,
            trajectory_evaluators=[my_evaluator]
        )

        # Run the simulation
        result = await simulator(initial_trajectory={"messages": [{"role": "user", "content": "Start"}]})
        ```
    """

    if max_turns is None and stopping_condition is None:
        raise ValueError(
            "At least one of max_turns or stopping_condition must be provided."
        )

    @traceable(name="multiturn_simulator")
    async def _run_simulator(
        *,
        initial_trajectory: MultiturnSimulatorTrajectory,
        reference_outputs: Optional[Any] = None,
        runnable_config: Optional[RunnableConfig] = None,
        **kwargs,
    ):
        turn_counter = 0
        current_reduced_trajectory: MultiturnSimulatorTrajectory = {"messages": []}
        wrapped_app = _wrap(app, "app")
        if isinstance(user, list):
            static_responses = user
            simulated_user = _create_static_simulated_user(static_responses)
        else:
            simulated_user = user  # type: ignore
        wrapped_simulated_user = _wrap(simulated_user, "simulated_user")
        while True:
            if max_turns is not None and turn_counter >= max_turns:
                break
            current_inputs = (
                initial_trajectory
                if turn_counter == 0
                else await wrapped_simulated_user.ainvoke(
                    current_reduced_trajectory, config=runnable_config
                )
            )
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory,
                current_inputs,
                update_source="user",
                turn_counter=turn_counter,
            )
            current_outputs = await wrapped_app.ainvoke(
                current_reduced_trajectory, config=runnable_config
            )
            current_reduced_trajectory = _trajectory_reducer(
                current_reduced_trajectory,
                current_outputs,
                update_source="app",
                turn_counter=turn_counter,
            )
            turn_counter += 1
            if stopping_condition and await stopping_condition(
                current_reduced_trajectory
            ):
                break
        results = []
        del current_reduced_trajectory["turn_counter"]
        for trajectory_evaluator in trajectory_evaluators or []:
            try:
                trajectory_eval_result = await trajectory_evaluator(
                    outputs=current_reduced_trajectory,
                    reference_outputs=reference_outputs,
                )
                if isinstance(trajectory_eval_result, list):
                    results.extend(trajectory_eval_result)
                else:
                    results.append(trajectory_eval_result)
            except Exception as e:
                print(f"Error in trajectory evaluator {trajectory_evaluator}: {e}")
        return MultiturnSimulatorResult(
            trajectory=current_reduced_trajectory,
            evaluator_results=results,
        )

    return _run_simulator
