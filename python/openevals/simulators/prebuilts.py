import uuid
from typing import Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from openevals.types import ChatCompletionMessage
from openevals.utils import _convert_to_openai_message


def _is_internal_message(message: ChatCompletionMessage) -> bool:
    return bool(
        message.get("role") != "user"
        and (message.get("role") != "assistant" or message.get("tool_calls"))
    )


def create_llm_simulated_user(
    *,
    system: str,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
    fixed_responses: Optional[list[Union[str, ChatCompletionMessage]]] = None,
):
    """Creates a simulated user powered by a language model for multi-turn conversations.

    This function generates a simulator that can be used with the run_multiturn_simulator method to create
    dynamic, LLM-powered user responses in a conversation. The simulator automatically handles message
    role conversion to maintain proper conversation flow, where user messages become assistant messages
    and vice versa when passed to the underlying LLM.

    Args:
        system: System prompt that guides the LLM's behavior as a simulated user.
        model: Optional name of the language model to use. Must be provided if client is not.
        client: Optional LangChain chat model instance. Must be provided if model is not.
        fixed_responses: Optional list of fixed responses to use for the simulated user.
            If the number of turns exceeds the number of fixed responses, the simulated user will
            generate a response using the specified LLM.
    Returns:
        A callable simulator function that takes a MultiturnSimulatorTrajectory containing conversation messages
        and returns a MultiturnSimulatorTrajectory with the simulated user's response.

    Example:
        ```python
        from openevals.simulators import run_multiturn_simulation, create_llm_simulated_user

        # Create a simulated user with GPT-4
        simulated_user = create_llm_simulated_user(
            system="You are a helpful customer service representative",
            model="openai:gpt-5-mini"
        )

        # Use with run_multiturn_simulation
        simulator = run_multiturn_simulation(
            app=my_chat_app,
            user=simulated_user,
            max_turns=5,
            thread_id="1"
        )
        ```

    Notes:
        - The simulator automatically converts message roles to maintain proper conversation flow:
          * User messages become assistant messages when sent to the LLM
          * Assistant messages (without tool calls) become user messages when sent to the LLM
        - The system prompt is prepended to each conversation to maintain consistent behavior
    """
    if not model and not client:
        raise ValueError("Either model or client must be provided")
    if model and client:
        raise ValueError("Only one of model or client must be provided")

    if not client:
        client = init_chat_model(model=model)  # type: ignore

    def _simulator(
        current_trajectory: list[ChatCompletionMessage],
        *,
        turn_counter: int,
        **kwargs,
    ):
        if fixed_responses and turn_counter < len(fixed_responses):
            res = fixed_responses[turn_counter]
            if isinstance(res, str):
                return {"role": "user", "content": res, "id": str(uuid.uuid4())}
            else:
                return res
        messages = []
        for msg in current_trajectory:
            converted_message = _convert_to_openai_message(msg)
            if _is_internal_message(converted_message):
                continue
            if converted_message.get("role") == "user":
                converted_message["role"] = "assistant"
                messages.append(converted_message)
            elif converted_message.get(
                "role"
            ) == "assistant" and not converted_message.get("tool_calls"):
                converted_message["role"] = "user"
                messages.append(converted_message)
        if len(messages) == 0:
            messages = [
                {
                    "role": "user",
                    "content": "Generate an initial query to start a conversation based on your instructions. Do not respond with other text.",
                    "id": str(uuid.uuid4()),
                }
            ]
        messages = [{"role": "system", "content": system}] + messages  # type: ignore
        response = client.invoke(messages)  # type: ignore
        return {"role": "user", "content": response.content, "id": response.id}

    return _simulator


def create_async_llm_simulated_user(
    *,
    system: str,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
    fixed_responses: Optional[list[Union[str, ChatCompletionMessage]]] = None,
):
    """Creates an async simulated user powered by a language model for multi-turn conversations.

    This function generates a simulator that can be used with the run_multiturn_simulation_async method to create
    dynamic, LLM-powered user responses in a conversation. The simulator automatically handles message
    role conversion to maintain proper conversation flow, where user messages become assistant messages
    and vice versa when passed to the underlying LLM.

    Args:
        system: System prompt that guides the LLM's behavior as a simulated user.
        model: Optional name of the language model to use. Must be provided if client is not.
        client: Optional LangChain chat model instance. Must be provided if model is not.
        fixed_responses: Optional list of fixed responses to use for the simulated user.
            If the number of turns exceeds the number of fixed responses, the simulated user will
            generate a response using the specified LLM.

    Returns:
        A callable simulator function that takes a MultiturnSimulatorTrajectory containing conversation messages
        and returns a MultiturnSimulatorTrajectory with the simulated user's response.

    Example:
        ```python
        from openevals.simulators import run_multiturn_simulation_async, create_async_llm_simulated_user

        # Create a simulated user with GPT-4
        simulated_user = create_async_llm_simulated_user(
            system="You are a helpful customer service representative",
            model="openai:gpt-5-mini"
        )

        # Use with run_multiturn_simulation_async
        simulator = run_multiturn_simulation_async(
            app=my_chat_app,
            user=simulated_user,
            max_turns=5,
            thread_id="1"
        )
        ```

    Notes:
        - The simulator automatically converts message roles to maintain proper conversation flow:
          * User messages become assistant messages when sent to the LLM
          * Assistant messages (without tool calls) become user messages when sent to the LLM
        - The system prompt is prepended to each conversation to maintain consistent behavior
    """
    if not model and not client:
        raise ValueError("Either model or client must be provided")
    if model and client:
        raise ValueError("Only one of model or client must be provided")

    if not client:
        client = init_chat_model(model=model)  # type: ignore

    async def _simulator(
        current_trajectory: list[ChatCompletionMessage],
        *,
        turn_counter: int,
        **kwargs,
    ):
        if fixed_responses and turn_counter < len(fixed_responses):
            res = fixed_responses[turn_counter]
            if isinstance(res, str):
                return {"role": "user", "content": res, "id": str(uuid.uuid4())}
            else:
                return res
        messages = []
        for msg in current_trajectory:
            converted_message = _convert_to_openai_message(msg)
            if _is_internal_message(converted_message):
                continue
            if converted_message.get("role") == "user":
                converted_message["role"] = "assistant"
                messages.append(converted_message)
            elif converted_message.get(
                "role"
            ) == "assistant" and not converted_message.get("tool_calls"):
                converted_message["role"] = "user"
                messages.append(converted_message)
        if len(messages) == 0:
            messages = [
                {
                    "role": "user",
                    "content": "Generate an initial query to start a conversation based on your instructions.",
                    "id": str(uuid.uuid4()),
                }
            ]
        messages = [{"role": "system", "content": system}] + messages  # type: ignore
        response = await client.ainvoke(messages)  # type: ignore
        return {"role": "user", "content": response.content, "id": response.id}

    return _simulator
