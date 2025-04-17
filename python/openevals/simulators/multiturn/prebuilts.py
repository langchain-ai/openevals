from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from openevals.simulators.multiturn.types import TrajectoryDict
from openevals.utils import _convert_to_openai_message


def create_llm_simulated_user(
    *,
    system: str,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
):
    if not model and not client:
        raise ValueError("Either model or client must be provided")
    if model and client:
        raise ValueError("Only one of model or client must be provided")

    if not client:
        client = init_chat_model(model=model)  # type: ignore

    def _simulator(inputs: TrajectoryDict):
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
