from __future__ import annotations

from langsmith import testing as t
from langsmith.testing._internal import _TEST_CASE
import functools
from typing import Any, Callable, TYPE_CHECKING, Union, Optional

__all__ = [
    "_chat_completion_messages_to_string",
    "_run_evaluator",
    "_arun_evaluator",
    "_normalize_to_openai_messages_list",
]

from openevals.types import ChatCompletionMessage, EvaluatorResult

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


@functools.lru_cache(maxsize=1)
def _import_langchain_core() -> tuple:
    from langchain_core.messages.utils import convert_to_openai_messages
    from langchain_core.messages import BaseMessage

    return BaseMessage, convert_to_openai_messages


def _convert_to_openai_message(
    message: ChatCompletionMessage | BaseMessage | dict,
) -> ChatCompletionMessage:
    if isinstance(message, dict):
        return message  # type: ignore
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
    messages: Optional[Union[list[ChatCompletionMessage], list[BaseMessage], dict]],
) -> list[ChatCompletionMessage]:
    if messages is None:
        return []
    if isinstance(messages, dict):
        if "messages" in messages:
            messages = messages["messages"]
        else:
            raise ValueError("if messages is a dict, it must contain a 'messages' key")
    return [_convert_to_openai_message(message) for message in messages]  # type: ignore


# Helper function to process individual scores
def _process_score(key: str, value: Any) -> tuple[float, str | None]:
    if isinstance(value, dict):
        if "score" in value:
            return value["score"], value.get("reasoning")
        raise ValueError(
            f"Expected a dictionary with keys 'score' and 'reasoning', but got {value}"
        )
    return value, None


def _run_evaluator(
    *, run_name: str, scorer: Callable, feedback_key: str, **kwargs: Any
) -> EvaluatorResult | list[EvaluatorResult]:
    def _run_scorer():
        # Get the initial score
        score = scorer(**kwargs)

        # Collect all results first
        if isinstance(score, dict):
            results = []
            # Handle dictionary of scores
            for key, value in score.items():
                key_score, reasoning = _process_score(key, value)
                results.append(
                    EvaluatorResult(key=key, score=key_score, comment=reasoning)
                )
            return results
        else:
            # Handle single score
            if isinstance(score, tuple):
                score, reasoning = score
            else:
                reasoning = None
            return EvaluatorResult(key=feedback_key, score=score, comment=reasoning)

    # Log feedback if in test case
    if _TEST_CASE.get():
        with t.trace_feedback(name=run_name):
            results = _run_scorer()
            if isinstance(results, list):
                for result in results:
                    t.log_feedback(
                        key=result["key"],
                        score=result["score"],
                        comment=result["comment"],
                    )
            else:
                t.log_feedback(
                    key=results["key"],
                    score=results["score"],
                    comment=results["comment"],
                )
    else:
        results = _run_scorer()

    # Return single result or list of results
    return results


async def _arun_evaluator(
    *, run_name: str, scorer: Callable, feedback_key: str, **kwargs: Any
) -> EvaluatorResult | list[EvaluatorResult]:
    async def _arun_scorer():
        # Get the initial score
        score = await scorer(**kwargs)

        # Collect all results first
        if isinstance(score, dict):
            results = []
            # Handle dictionary of scores
            for key, value in score.items():
                key_score, reasoning = _process_score(key, value)
                results.append(
                    EvaluatorResult(key=key, score=key_score, comment=reasoning)
                )
            return results
        else:
            # Handle single score
            if isinstance(score, tuple):
                score, reasoning = score
            else:
                reasoning = None
            return EvaluatorResult(key=feedback_key, score=score, comment=reasoning)

    # Log feedback if in test case
    if _TEST_CASE.get():
        with t.trace_feedback(name=run_name):
            results = await _arun_scorer()
            if isinstance(results, list):
                for result in results:
                    t.log_feedback(
                        key=result["key"],
                        score=result["score"],
                        comment=result["comment"],
                    )
            else:
                t.log_feedback(
                    key=results["key"],
                    score=results["score"],
                    comment=results["comment"],
                )
    else:
        results = await _arun_scorer()

    # Return single result or list of results
    return results


def _chat_completion_messages_to_string(messages: list[ChatCompletionMessage]) -> str:
    def format_message(message: ChatCompletionMessage) -> str:
        content = message.get("content", "")  # Handle None content

        # Handle tool/function calls
        tool_calls = message.get("tool_calls") or []
        if message.get("tool_calls", None):
            tool_calls_str = "\n".join(
                f"<tool_call>\n"
                f"<name>{call.get('function', {}).get('name', '')}</name>\n"
                f"<arguments>{call.get('function', {}).get('arguments', '')}</arguments>\n"
                f"</tool_call>"
                for call in tool_calls
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
