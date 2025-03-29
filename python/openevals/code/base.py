from openevals.types import (
    ScoreType,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
)

from openevals.utils import (
    _normalize_final_app_outputs_as_string,
    _run_evaluator,
    _arun_evaluator,
)

from typing import Any, Literal, Union, Optional, Callable, Awaitable
from typing_extensions import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model

__all__ = [
    "_create_base_code_evaluator",
    "_create_async_base_code_evaluator",
]

LLM_EXTRACTION_SYSTEM_PROMPT = """
You are an expert software auditor.

<Instructions>
  Your job is to extract code from a given text.

  - If there is code - extract it into a single script by calling the provided "ExtractCode" tool.
  - If there is no code to extract - call "NoCode".

  If you extract code, your response will be passed DIRECTLY into a code execution sandbox for further testing,
  so make sure to extract all code **without modifications**, even if it contains errors,
  since any modifications will ruin the integrity of the testing process.
  Omit installation instructions and shell commands from any code you extract.
</Instructions>
"""

LLM_EXTRACTION_USER_PROMPT = """
Extract code from the following:

<text>
{outputs}
</text>
"""


class ExtractCode(TypedDict):
    """Tool to call if there is code to extract.
    Omit installation instructions and shell commands."""

    code: str


class NoCode(TypedDict):
    """Tool to call to indicate no code was found."""

    no_code: bool


def _extract_code_from_markdown_code_blocks(text: str) -> Optional[str]:
    """
    Extract code from markdown code blocks in the provided text.

    Supports both triple backtick code blocks with or without language specifiers.

    Args:
        text: The text containing markdown code blocks

    Returns:
        A string containing only the code extracted from code blocks, with blocks
        separated by newlines
    """
    import re

    # Pattern to match code blocks with or without language specifier
    # (?s) enables dot to match newlines
    # (?:```(?:\w+)?\n(.*?)```) matches code blocks with optional language specifier
    pattern = r"(?m)^(?<!`)\`\`\`(\w*)\n([\s\S]*?)^(?<!`)\`\`\`$"

    # Find all code blocks
    matches = re.finditer(pattern, text, re.MULTILINE)

    # Filter out bash/shell blocks and collect valid code blocks
    excluded_langs = {
        "bash",
        "sh",
        "shell",
        "zsh",
        "fish",
        "console",
        "terminal",
        "json",
    }
    code_blocks = []
    for match in matches:
        lang = match.group(1).strip()
        if lang not in excluded_langs:
            code_blocks.append(match.group(2))

    if not code_blocks:
        return None  # Return None if no code blocks found

    # Join all code blocks with newlines
    return "\n".join(code_blocks)


def _create_base_code_evaluator(
    *,
    scorer: Callable[..., Union[ScoreType, tuple[bool, Optional[str], Optional[dict]]]],
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
    run_name: str,
    feedback_key: str,
) -> SimpleEvaluator:
    if code_extractor is not None and code_extraction_strategy != "none":
        raise ValueError(
            "`code_extractor` and `code_extraction_strategy` cannot both be provided"
        )
    if code_extraction_strategy == "llm":
        if model is None and client is None:
            raise ValueError("You must provide either a `model` string or a `client`")
        if client is None:
            client = init_chat_model(model)  # type: ignore

    def _wrapped_evaluator(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Union[str, dict],
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ):
        def _score_wrapper(*, outputs: Union[str, dict], **kwargs):
            if code_extractor is None:
                normalized_outputs = _normalize_final_app_outputs_as_string(outputs)
                if code_extraction_strategy == "llm":
                    model_with_tools = client.bind_tools([ExtractCode, NoCode])  # type: ignore
                    res = model_with_tools.invoke(
                        [
                            {"role": "system", "content": LLM_EXTRACTION_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": LLM_EXTRACTION_USER_PROMPT.format(
                                    outputs=normalized_outputs
                                ),
                            },
                        ],
                        {"run_name": "extract_code"},
                    )
                    if res.tool_calls[0]["name"] == "ExtractCode":  # type: ignore
                        normalized_outputs = res.tool_calls[0]["args"]["code"]  # type: ignore
                    else:
                        return (False, None, {"code_extraction_failed": True})
                elif code_extraction_strategy == "markdown_code_blocks":
                    normalized_outputs = _extract_code_from_markdown_code_blocks(  # type: ignore
                        normalized_outputs
                    )
                    if normalized_outputs is None:
                        return (False, None, {"code_extraction_failed": True})
                else:
                    # Nothing to do to extract code
                    pass
            else:
                normalized_outputs = code_extractor(outputs)
            return scorer(
                outputs=normalized_outputs,
                **kwargs,
            )

        return _run_evaluator(
            run_name=run_name,
            scorer=_score_wrapper,
            feedback_key=feedback_key,
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator


def _create_async_base_code_evaluator(
    *,
    scorer: Callable[..., Union[ScoreType, Awaitable[ScoreType]]],
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], Union[str, Awaitable[str]]]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
    run_name: str,
    feedback_key: str,
) -> SimpleAsyncEvaluator:
    if code_extractor is not None and code_extraction_strategy != "none":
        raise ValueError(
            "`code_extractor` and `code_extraction_strategy` cannot both be provided"
        )

    if code_extraction_strategy == "llm":
        if model is None and client is None:
            raise ValueError("You must provide either a `model` string or a `client`")
        if client is None:
            client = init_chat_model(model)  # type: ignore

    async def _wrapped_evaluator(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Union[str, dict],
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ):
        async def _ascore_wrapper(*, outputs: Union[str, dict], **kwargs):
            if code_extractor is None:
                normalized_outputs = _normalize_final_app_outputs_as_string(outputs)
                if code_extraction_strategy == "llm":
                    model_with_tools = client.bind_tools([ExtractCode, NoCode])  # type: ignore
                    res = await model_with_tools.ainvoke(
                        [
                            {"role": "system", "content": LLM_EXTRACTION_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": LLM_EXTRACTION_USER_PROMPT.format(
                                    outputs=normalized_outputs
                                ),
                            },
                        ],
                        {"run_name": "extract_code"},
                    )
                    if res.tool_calls[0]["name"] == "ExtractCode":  # type: ignore
                        normalized_outputs = res.tool_calls[0]["args"]["code"]  # type: ignore
                    else:
                        return (False, None, {"code_extraction_failed": True})
                elif code_extraction_strategy == "markdown_code_blocks":
                    normalized_outputs = _extract_code_from_markdown_code_blocks(  # type: ignore
                        normalized_outputs
                    )
                    if normalized_outputs is None:
                        return (False, None, {"code_extraction_failed": True})
                else:
                    # Nothing to do to extract code
                    pass
            else:
                normalized_outputs = code_extractor(outputs)  # type: ignore
                if hasattr(normalized_outputs, "__await__"):
                    normalized_outputs = await normalized_outputs
            score_result = scorer(
                outputs=normalized_outputs,
                **kwargs,
            )
            if hasattr(score_result, "__await__"):
                return await score_result
            return score_result

        return await _arun_evaluator(
            run_name=run_name,
            scorer=_ascore_wrapper,
            feedback_key=feedback_key,
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator  # type: ignore
