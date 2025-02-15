from __future__ import annotations


from langsmith import testing as t
from langsmith.testing._internal import _TEST_CASE
import functools
from typing import Any, Callable, TYPE_CHECKING, Union

__all__ = [
    "_chat_completion_messages_to_string",
    "_run_evaluator",
    "_normalize_to_openai_messages_list",
]

from evaluators.types import ChatCompletionMessage, EvaluatorResult

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


@functools.lru_cache(maxsize=1)
def _import_langchain_core() -> tuple:
    from langchain_core.messages.utils import convert_to_openai_messages
    from langchain_core.messages import BaseMessage

    return BaseMessage, convert_to_openai_messages


def _convert_to_openai_message(message: BaseMessage | dict) -> dict:
    if isinstance(message, dict):
        return message
    else:
        try:
            BaseMessage, convert_to_openai_messages = _import_langchain_core()
        except ImportError:
            raise ValueError(
                "Only messages in OpenAI format or LangChain BaseMessage are supported. If not passing messages in OpenAI format, you must install `langchain_core`."
            )
        if not isinstance(message, BaseMessage):
            raise ValueError(f"Expected BaseMessage, got {type(message)}")
        return convert_to_openai_messages([message])[0]


def _normalize_to_openai_messages_list(
    messages: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
) -> list[dict]:
    if isinstance(messages, dict):
        if "messages" in messages:
            messages = messages["messages"]
        else:
            raise ValueError("if messages is a dict, it must contain a 'messages' key")
    return [_convert_to_openai_message(message) for message in messages]


def _run_evaluator(
    *, run_name: str, scorer: Callable, feedback_key: str, **kwargs: Any
) -> EvaluatorResult | list[EvaluatorResult]:
    # Get the initial score
    score = scorer(**kwargs)
    
    # Helper function to process individual scores
    def process_score(key: str, value: Any) -> tuple[float, str | None]:
        if isinstance(value, dict):
            if set(value.keys()) == {"score", "reasoning"}:
                return value["score"], value["reasoning"]
            raise ValueError(
                f"Expected a dictionary with keys 'score' and 'reasoning', but got {value}"
            )
        return value, None

    # Collect all results first
    results = []
    if isinstance(score, dict):
        # Handle dictionary of scores
        for key, value in score.items():
            key_score, reasoning = process_score(key, value)
            results.append(EvaluatorResult(key=key, score=key_score, comment=reasoning))
    else:
        # Handle single score
        if isinstance(score, tuple):
            score, reasoning = score
        else:
            reasoning = None
        results.append(EvaluatorResult(key=feedback_key, score=score, comment=reasoning))

    # Log feedback if in test case
    if _TEST_CASE.get():
        with t.trace_feedback(name=run_name):
            for result in results:
                t.log_feedback(
                    key=result["key"],
                    score=result["score"],
                    comment=result["comment"]
                )

    # Return single result or list of results
    return results[0] if len(results) == 1 else results


def _chat_completion_messages_to_string(messages: list[ChatCompletionMessage]) -> str:
    def format_message(message: ChatCompletionMessage) -> str:
        content = message.get("content", "")  # Handle None content

        # Handle tool/function calls
        if message.get("tool_calls", None):
            tool_calls_str = "\n".join(
                f"<tool_call>\n"
                f"<name>{call.get('function', {}).get('name', '')}</name>\n"
                f"<arguments>{call.get('function', {}).get('arguments', '')}</arguments>\n"
                f"</tool_call>"
                for call in message.get("tool_calls")
            )
            content = f"{content}\n{tool_calls_str}" if content else tool_calls_str

        # Handle tool call results
        if message.get("tool_call_id", None):
            content = (
                f"<tool_result>\n"
                f"<id>{message.get('tool_call_id')}</id>\n"
                f"<content>{content}</content>\n"
                f"</tool_result>"
            )

        return f"<{message.get('role', '')}>\n{content}\n</{message.get('role', '')}>"

    return "\n\n".join(format_message(message) for message in messages)
