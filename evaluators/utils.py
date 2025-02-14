__all__ = ["_chat_completion_messages_to_string", "_run_evaluator"]

from langsmith import testing as t
from langsmith.testing._internal import _TEST_CASE
from typing import Any, Callable

from evaluators.types import ChatCompletionMessage, EvaluatorResult


def _run_evaluator(
    *, run_name: str, scorer: Callable, feedback_key: str, **kwargs: Any
) -> EvaluatorResult | list[EvaluatorResult]:
    reasoning = None
    score = scorer(**kwargs)
    if _TEST_CASE.get():
        with t.trace_feedback(name=run_name):
            if isinstance(score, dict):
                for individual_feedback_key in score:
                    if isinstance(score[individual_feedback_key], dict):
                        if set(score[individual_feedback_key].keys()) == {"score", "reasoning"}:
                            key_score = score[individual_feedback_key]["score"]
                            reasoning = score[individual_feedback_key]["reasoning"]
                        else:
                            raise ValueError(
                                f"Expected a dictionary with keys 'score' and 'reasoning', but got {score[individual_feedback_key]}"
                            )
                    else:
                        key_score = score[individual_feedback_key]
                        reasoning = None
                    t.log_feedback(key=individual_feedback_key, score=key_score, comment=reasoning)
            else:
                if isinstance(score, tuple):
                    score, reasoning = score
                t.log_feedback(key=feedback_key, score=score, comment=reasoning)
    # Always return the feedback, even if not logging
    if isinstance(score, dict):
        results = []
        for individual_feedback_key in score:
            if isinstance(score[individual_feedback_key], dict):
                if set(score[individual_feedback_key].keys()) == {"score", "reasoning"}:
                    key_score = score[individual_feedback_key]["score"]
                    reasoning = score[individual_feedback_key]["reasoning"]
                else:
                    raise ValueError(
                        f"Expected a dictionary with keys 'score' and 'reasoning', but got {score[individual_feedback_key]}"
                    )
            else:
                key_score = score[individual_feedback_key]
                reasoning = None
            results.append(EvaluatorResult(key=individual_feedback_key, score=key_score, comment=reasoning))
        return results
    else:
        if isinstance(score, tuple):
            score, reasoning = score
        return EvaluatorResult(key=feedback_key, score=score, comment=reasoning)


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
