from __future__ import annotations
import asyncio

from langsmith import testing as t, get_current_run_tree, traceable
from langsmith.testing._internal import _TEST_CASE
from typing import Any, Callable, Union, Optional

from openevals.types import ChatCompletionMessage, EvaluatorResult

from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.messages import BaseMessage

__all__ = [
    "_chat_completion_messages_to_string",
    "_run_evaluator",
    "_arun_evaluator",
    "_normalize_to_openai_messages_list",
    "_normalize_final_app_outputs_as_string",
]


def _convert_to_openai_message(
    message: Union[ChatCompletionMessage, BaseMessage, dict],
) -> ChatCompletionMessage:
    if not isinstance(message, BaseMessage) and not isinstance(message, dict):
        message = dict(message)
    converted = convert_to_openai_messages([message])[0]  # type: ignore
    if isinstance(message, BaseMessage):
        if message.id is not None and converted.get("id") is None:
            converted["id"] = message.id
    else:
        if message.get("id") is not None and converted.get("id") is None:
            converted["id"] = message.get("id")
    return converted  # type: ignore


def _normalize_to_openai_messages_list(
    messages: Optional[
        Union[
            list[ChatCompletionMessage], list[BaseMessage], ChatCompletionMessage, dict
        ]
    ],
) -> list[ChatCompletionMessage]:
    if messages is None:
        return []
    if isinstance(messages, dict):
        if "role" in messages:
            messages = [messages]  # type: ignore
        elif "messages" in messages:
            messages = messages["messages"]  # type: ignore
        else:
            raise ValueError("if messages is a dict, it must contain a 'messages' key")
    if not isinstance(messages, list):
        messages = [messages]  # type: ignore
    return [_convert_to_openai_message(message) for message in messages]  # type: ignore


# Helper function to process individual scores
def _process_score(
    key: str, value: Any
) -> tuple[float, Union[str, None], Union[dict, None], Optional[str]]:
    if isinstance(value, dict):
        if "score" in value:
            return (
                value["score"],
                value.get("reasoning"),
                value.get("metadata", None),
                value.get("source_run_id", None),
            )  # type: ignore
        raise ValueError(
            f"Expected a dictionary with keys 'score' and 'reasoning', but got {value}"
        )
    return value, None, None, None


def _add_metadata_and_inputs_to_run_tree(
    run_name: str,
    framework: Union[str, None] = None,
    results: Optional[Union[dict, list[dict]]] = None,
    inputs: Optional[Any] = None,
):
    rt = get_current_run_tree()
    if rt is not None:
        if results is not None:
            if isinstance(results, list):
                for result in results:
                    if result.get("metadata", None) is not None:
                        rt.metadata.update(result.get("metadata", None))
            else:
                try:
                    if results.get("metadata", None) is not None:
                        rt.metadata.update(results.get("metadata", None))
                except Exception:
                    pass
            if inputs is not None:
                rt.inputs = inputs
        rt.metadata["__ls_framework"] = framework
        rt.metadata["__ls_evaluator"] = run_name
        rt.metadata["__ls_language"] = "python"


def _run_evaluator(
    *,
    run_name: str,
    scorer: Callable,
    feedback_key: str,
    ls_framework: str = "openevals",
    **kwargs: Any,
) -> Union[EvaluatorResult, list[EvaluatorResult]]:
    return _run_evaluator_untyped(  # type: ignore
        run_name=run_name,
        scorer=scorer,
        feedback_key=feedback_key,
        return_raw_outputs=False,
        ls_framework=ls_framework,
        **kwargs,
    )


def _run_evaluator_untyped(
    *,
    run_name: str,
    scorer: Callable,
    feedback_key: str,
    return_raw_outputs: bool = False,
    ls_framework: str = "openevals",
    **kwargs: Any,
) -> Union[EvaluatorResult, list[EvaluatorResult], dict]:
    @traceable(name=run_name)
    def _run_scorer(**kwargs: Any):
        # Get the initial score
        score = scorer(**kwargs)

        if return_raw_outputs:
            return score

        # Collect all results first
        if isinstance(score, dict):
            results = []
            # Handle dictionary of scores
            for key, value in score.items():
                if isinstance(value, list):
                    for item in value:
                        key_score, reasoning, metadata, source_run_id = _process_score(
                            key, item
                        )
                        result = EvaluatorResult(
                            key=key,
                            score=key_score,
                            comment=reasoning,
                            metadata=metadata,
                        )
                        if source_run_id is not None:
                            result["source_run_id"] = source_run_id
                        results.append(result)
                else:
                    key_score, reasoning, metadata, source_run_id = _process_score(
                        key, value
                    )
                    result = EvaluatorResult(
                        key=key,
                        score=key_score,
                        comment=reasoning,
                        metadata=metadata,
                    )
                    if source_run_id is not None:
                        result["source_run_id"] = source_run_id
                    results.append(result)
            return results
        else:
            # Handle single score
            if isinstance(score, tuple):
                if len(score) == 3:
                    score, reasoning, metadata = score
                elif len(score) == 2:
                    score, reasoning = score
                    metadata = None
                else:
                    raise ValueError(f"Expected a tuple of length 2 or 3, got {score}")
            else:
                reasoning = None
                metadata = None
            return EvaluatorResult(
                key=feedback_key, score=score, comment=reasoning, metadata=metadata
            )

    # Log feedback if in test case
    if _TEST_CASE.get():
        with t.trace_feedback(name=run_name):
            results = _run_scorer(**kwargs)

            _add_metadata_and_inputs_to_run_tree(
                run_name,
                ls_framework,
                results,
                inputs={
                    "inputs": kwargs.get("inputs", None),
                    "reference_outputs": kwargs.get("reference_outputs", None),
                }
                if kwargs.get("inputs", None) is not None
                or kwargs.get("reference_outputs", None) is not None
                else None,
            )
            if not return_raw_outputs:
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
        results = _run_scorer(**kwargs)
        _add_metadata_and_inputs_to_run_tree(run_name, ls_framework, results)

    # Return single result or list of results
    return results


async def _arun_evaluator(
    *,
    run_name: str,
    scorer: Callable,
    feedback_key: str,
    return_raw_outputs: bool = False,
    ls_framework: str = "openevals",
    **kwargs: Any,
) -> Union[EvaluatorResult, list[EvaluatorResult]]:
    return await _arun_evaluator_untyped(  # type: ignore
        run_name=run_name,
        scorer=scorer,
        feedback_key=feedback_key,
        return_raw_outputs=return_raw_outputs,
        ls_framework=ls_framework,
        **kwargs,
    )


async def _arun_evaluator_untyped(
    *,
    run_name: str,
    scorer: Callable,
    feedback_key: str,
    return_raw_outputs: bool = False,
    ls_framework: str = "openevals",
    **kwargs: Any,
) -> Union[EvaluatorResult, list[EvaluatorResult], dict]:
    @traceable(name=run_name)
    async def _arun_scorer(**kwargs: Any):
        # Get the initial score
        if asyncio.iscoroutinefunction(scorer):
            score = await scorer(**kwargs)
        else:
            score = scorer(**kwargs)

        if return_raw_outputs:
            return score

        # Collect all results first
        if isinstance(score, dict):
            results = []
            # Handle dictionary of scores
            for key, value in score.items():
                if isinstance(value, list):
                    for item in value:
                        key_score, reasoning, metadata, source_run_id = _process_score(
                            key, item
                        )
                        result = EvaluatorResult(
                            key=key,
                            score=key_score,
                            comment=reasoning,
                            metadata=metadata,
                        )
                        if source_run_id is not None:
                            result["source_run_id"] = source_run_id
                        results.append(result)
                else:
                    key_score, reasoning, metadata, source_run_id = _process_score(
                        key, value
                    )
                    result = EvaluatorResult(
                        key=key,
                        score=key_score,
                        comment=reasoning,
                        metadata=metadata,
                    )
                    if source_run_id is not None:
                        result["source_run_id"] = source_run_id
                    results.append(result)
            return results
        else:
            # Handle single score
            if isinstance(score, tuple):
                if len(score) == 3:
                    score, reasoning, metadata = score
                elif len(score) == 2:
                    score, reasoning = score
                    metadata = None
                else:
                    raise ValueError(f"Expected a tuple of length 2 or 3, got {score}")
            else:
                reasoning = None
                metadata = None
            return EvaluatorResult(
                key=feedback_key, score=score, comment=reasoning, metadata=metadata
            )

    # Log feedback if in test case
    if _TEST_CASE.get():
        with t.trace_feedback(name=run_name):
            results = await _arun_scorer(**kwargs)
            _add_metadata_and_inputs_to_run_tree(
                run_name,
                ls_framework,
                results,
                inputs={
                    "inputs": kwargs.get("inputs", None),
                    "reference_outputs": kwargs.get("reference_outputs", None),
                }
                if kwargs.get("inputs", None) is not None
                or kwargs.get("reference_outputs", None) is not None
                else None,
            )
            if not return_raw_outputs:
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
        results = await _arun_scorer(**kwargs)
        _add_metadata_and_inputs_to_run_tree(run_name, ls_framework, results)

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


def _normalize_final_app_outputs_as_string(outputs: Union[str, dict]):
    if isinstance(outputs, str):
        return outputs
    elif isinstance(outputs, dict):
        if "content" in outputs:
            converted_message = _convert_to_openai_message(outputs)
            return converted_message["content"]
        elif "messages" in outputs and isinstance(outputs["messages"], list):
            final_message = _convert_to_openai_message(outputs["messages"][-1])
            return final_message["content"]
        else:
            raise ValueError(
                f"Expected a string, dictionary with a 'content' key or a 'messages' key with a list of messages, but got {outputs}"
            )
    else:
        raise ValueError(f"Expected str or dict, got {type(outputs)}")
