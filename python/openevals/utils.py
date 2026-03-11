from __future__ import annotations
import asyncio

from langsmith import testing as t, get_current_run_tree, traceable
from langsmith.testing._internal import _TEST_CASE
from typing import Any, Callable, Union, Optional

from openevals.types import ChatCompletionMessage, EvaluatorResult

from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.messages import BaseMessage, HumanMessage

__all__ = [
    "_chat_completion_messages_to_string",
    "_run_evaluator",
    "_arun_evaluator",
    "_normalize_to_openai_messages_list",
    "_normalize_final_app_outputs_as_string",
    "_attachment_to_content_block",
    "_normalize_content_blocks",
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


def _normalize_attachment_mime_type(mime_type: str) -> str:
    """Normalize MIME aliases to canonical forms."""
    normalized = mime_type.lower().strip()
    if normalized == "audio/mpeg":
        return "audio/mp3"
    if normalized in {"audio/wave", "audio/x-wav"}:
        return "audio/wav"
    return normalized


def _normalize_content_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize content blocks via LangChain's canonical form for cross-provider compatibility."""
    try:
        normalized = HumanMessage(content=blocks).content_blocks  # type: ignore[attr-defined]
        return [b for b in normalized if isinstance(b, dict)]
    except Exception:
        return blocks


def _attachment_to_content_block(item: Any) -> dict[str, Any]:
    """Convert an attachment to a content block dict for multimodal messages.

    Attachments should be passed in the multimodal trace format described at
    https://docs.langchain.com/langsmith/log-multimodal-traces:
    ``{"mime_type": "image/png", "data": "data:image/png;base64,..."}``.

    Also accepts plain image URL strings or pre-formatted content block dicts.

    Supported MIME types:
    - ``image/*``: ``{"type": "image_url", "image_url": {"url": data}}``
    - ``application/pdf``: ``{"type": "file", "file": {"filename": ..., "file_data": data}}``
    - ``audio/*``: ``{"type": "input_audio", "input_audio": {"data": base64, "format": fmt}}``
    """
    if isinstance(item, str):
        return {"type": "image_url", "image_url": {"url": item}}
    if not isinstance(item, dict):
        msg = (
            f"Unsupported attachment type: {type(item)}. "
            "Expected a string URL or a dict with 'mime_type' and 'data' keys."
        )
        raise ValueError(msg)

    mime_type = item.get("mime_type")
    data = item.get("data")

    if mime_type is None or data is None:
        if "type" not in item:
            msg = (
                "Attachment dict must contain either 'mime_type' and 'data' keys, "
                "or a 'type' key for pre-formatted content blocks."
            )
            raise ValueError(msg)
        return item

    mime_type = _normalize_attachment_mime_type(mime_type)

    if mime_type.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": data}}
    if mime_type == "application/pdf":
        filename = item.get("name", "attachment.pdf")
        return {"type": "file", "file": {"filename": filename, "file_data": data}}
    if mime_type.startswith("audio/"):
        base64_data = data.split(",")[1] if data.startswith("data:") else data
        fmt = mime_type.split("/")[1]
        return {"type": "input_audio", "input_audio": {"data": base64_data, "format": fmt}}
    msg = (
        f"Unsupported attachment MIME type: {mime_type}. "
        "Supported types: image/*, application/pdf, audio/*"
    )
    raise ValueError(msg)


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
