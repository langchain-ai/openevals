from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from openevals.types import MultiturnSimulatorTrajectory
from openevals.utils import _convert_to_openai_message


def create_llm_simulated_user(
    *,
    system: str,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
):
    """Creates a simulated user powered by a language model for multi-turn conversations.

    This function generates a simulator that can be used with the create_multiturn_simulator to create
    dynamic, LLM-powered user responses in a conversation. The simulator automatically handles message
    role conversion to maintain proper conversation flow, where user messages become assistant messages
    and vice versa when passed to the underlying LLM.

    Args:
        system: System prompt that guides the LLM's behavior as a simulated user.
        model: Optional name of the language model to use. Must be provided if client is not.
        client: Optional LangChain chat model instance. Must be provided if model is not.

    Returns:
        A callable simulator function that takes a MultiturnSimulatorTrajectory containing conversation messages
        and returns a MultiturnSimulatorTrajectory with the simulated user's response.

    Example:
        ```python
        from openevals.simulators import create_multiturn_simulator, create_llm_simulated_user

        # Create a simulated user with GPT-4
        simulated_user = create_llm_simulated_user(
            system="You are a helpful customer service representative",
            model="openai:gpt-4.1-mini"
        )

        # Use with create_multiturn_simulator
        simulator = create_multiturn_simulator(
            app=my_chat_app,
            user=simulated_user,
            max_turns=5
        )
        ```

    Notes:
        - The simulator automatically converts message roles to maintain proper conversation flow:
          * User messages become assistant messages when sent to the LLM
          * Assistant messages (without tool calls) become user messages when sent to the LLM
        - The system prompt is prepended to each conversation to maintain consistent behavior
        - The simulator returns responses in the format expected by create_multiturn_simulator
    """
    if not model and not client:
        raise ValueError("Either model or client must be provided")
    if model and client:
        raise ValueError("Only one of model or client must be provided")

    if not client:
        client = init_chat_model(model=model)  # type: ignore

    def _simulator(inputs: MultiturnSimulatorTrajectory):
        if not isinstance(inputs, dict) or not inputs["messages"]:
            raise ValueError(
                "Simulated user inputs must be a dict with a 'messages' key containing a list of messages"
            )
        messages = []
        for msg in inputs["messages"]:
            converted_message = _convert_to_openai_message(msg)
            if converted_message.get("role") == "user":
                converted_message["role"] = "assistant"
                messages.append(converted_message)
            elif converted_message.get(
                "role"
            ) == "assistant" and not converted_message.get("tool_calls"):
                converted_message["role"] = "user"
                messages.append(converted_message)
        if system:
            messages = [{"role": "system", "content": system}] + messages  # type: ignore
        response = client.invoke(messages)  # type: ignore
        return {
            "messages": [
                {"role": "user", "content": response.content, "id": response.id}
            ]
        }

    return _simulator


def create_async_llm_simulated_user(
    *,
    system: str,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
):
    """Creates an async simulated user powered by a language model for multi-turn conversations.

    This function generates a simulator that can be used with the create_async_multiturn_simulator to create
    dynamic, LLM-powered user responses in a conversation. The simulator automatically handles message
    role conversion to maintain proper conversation flow, where user messages become assistant messages
    and vice versa when passed to the underlying LLM.

    Args:
        system: System prompt that guides the LLM's behavior as a simulated user.
        model: Optional name of the language model to use. Must be provided if client is not.
        client: Optional LangChain chat model instance. Must be provided if model is not.

    Returns:
        A callable simulator function that takes a MultiturnSimulatorTrajectory containing conversation messages
        and returns a MultiturnSimulatorTrajectory with the simulated user's response.

    Example:
        ```python
        from openevals.simulators import create_async_multiturn_simulator, create_async_llm_simulated_user

        # Create a simulated user with GPT-4
        simulated_user = create_async_llm_simulated_user(
            system="You are a helpful customer service representative",
            model="openai:gpt-4.1-mini"
        )

        # Use with create_async_multiturn_simulator
        simulator = create_async_multiturn_simulator(
            app=my_chat_app,
            user=simulated_user,
            max_turns=5
        )
        ```

    Notes:
        - The simulator automatically converts message roles to maintain proper conversation flow:
          * User messages become assistant messages when sent to the LLM
          * Assistant messages (without tool calls) become user messages when sent to the LLM
        - The system prompt is prepended to each conversation to maintain consistent behavior
        - The simulator returns responses in the format expected by create_async_multiturn_simulator
    """
    if not model and not client:
        raise ValueError("Either model or client must be provided")
    if model and client:
        raise ValueError("Only one of model or client must be provided")

    if not client:
        client = init_chat_model(model=model)  # type: ignore

    async def _simulator(inputs: MultiturnSimulatorTrajectory):
        if not isinstance(inputs, dict) or not inputs["messages"]:
            raise ValueError(
                "Simulated user inputs must be a dict with a 'messages' key containing a list of messages"
            )
        messages = []
        for msg in inputs["messages"]:
            converted_message = _convert_to_openai_message(msg)
            if converted_message.get("role") == "user":
                converted_message["role"] = "assistant"
                messages.append(converted_message)
            elif converted_message.get(
                "role"
            ) == "assistant" and not converted_message.get("tool_calls"):
                converted_message["role"] = "user"
                messages.append(converted_message)
        if system:
            messages = [{"role": "system", "content": system}] + messages  # type: ignore
        response = await client.ainvoke(messages)  # type: ignore
        return {
            "messages": [
                {"role": "user", "content": response.content, "id": response.id}
            ]
        }

    return _simulator
