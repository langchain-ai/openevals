__all__ = ["_chat_completion_messages_to_string", "_run_evaluator"]

from langsmith import testing as t
from langsmith.testing._internal import _TEST_CASE
from typing import Any, Callable

from evaluators.types import ChatCompletionMessage, EvaluatorResult


def _run_evaluator(
    *, run_name: str, scorer: Callable, feedback_key: str, **kwargs: Any
) -> EvaluatorResult:
    reasoning = None
    if _TEST_CASE.get():
        with t.trace_feedback(name=run_name):
            score = scorer(**kwargs)
            if isinstance(score, tuple):
                score, reasoning = score
            t.log_feedback(key=feedback_key, score=score, comment=reasoning)
    else:
        score = scorer(**kwargs)
    return EvaluatorResult(key=feedback_key, score=score, comment=reasoning)


def _chat_completion_messages_to_string(messages: list[ChatCompletionMessage]) -> str:
    def format_message(message: ChatCompletionMessage) -> str:
        content = message.content or ""  # Handle None content

        # Handle tool/function calls
        if message.tool_calls:
            tool_calls_str = "\n".join(
                f"<tool_call>\n"
                f"<name>{call.function.name}</name>\n"
                f"<arguments>{call.function.arguments}</arguments>\n"
                f"</tool_call>"
                for call in message.tool_calls
            )
            content = f"{content}\n{tool_calls_str}" if content else tool_calls_str

        # Handle tool call results
        if message.tool_call_id:
            content = (
                f"<tool_result>\n"
                f"<id>{message.tool_call_id}</id>\n"
                f"<content>{content}</content>\n"
                f"</tool_result>"
            )

        return f"<{message.role}>\n{content}\n</{message.role}>"

    return "\n\n".join(format_message(message) for message in messages)
