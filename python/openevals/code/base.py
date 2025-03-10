from openevals.types import ScoreType, SimpleEvaluator, SimpleAsyncEvaluator

from openevals.utils import (
    _normalize_final_app_outputs_as_string,
    _run_evaluator,
    _arun_evaluator,
)

from typing import Any, Literal, Union, Optional, Callable, Awaitable

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
  Your response will be passed DIRECTLY into a code execution sandbox for further testing,
  so make sure to extract all code **without modifications**, even if it contains errors,
  since any modifications will ruin the integrity of the testing process.
  Do not respond with any text other than the code you extract.
</Instructions>

<Examples>
  <Example>
    <Input>
      <text>
        Here is some perfectly written Python code:
        
        ```python
        console.log("Hello, world!")
        ```
      </text>
    </Input>
    <Output>
      console.log("Hello, world!")
    </Output>
  </Example>
  <Example>
    <Input>
      <text>
        The first thing you should do is import numpy:
        
        ```
        import numpy as np
        ```
        
        Then, you should use numpy to create an array:
        
        ```
        arr = np.array([1, 2, 3, 4, 5])
        ```
        
        And finally, you should print the array:
        
        ```
        print(arr)
        ```
        
      </text>
    </Input>
    <Output>
      import numpy as np
      arr = np.array([1, 2, 3, 4, 5])
      print(arr)
    </Output>
  </Example>
</Examples>
"""

LLM_EXTRACTION_USER_PROMPT = """
Extract code from the following text:

<text>
{outputs}
</text>
"""


def _extract_code_from_markdown_code_blocks(text: str) -> str:
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
    pattern = r"(?s)```(?:\w+)?\n(.*?)```"

    # Find all code blocks
    code_blocks = re.findall(pattern, text)

    if not code_blocks:
        return text  # Return original text if no code blocks found

    # Join all code blocks with newlines
    return "\n".join(code_blocks)


def _create_base_code_evaluator(
    *,
    scorer: Callable[..., ScoreType],
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
            raise ValueError("Either model or client must be provided")
        if client is None:
            client = init_chat_model(model)  # type: ignore

    def _wrapped_evaluator(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Union[str, dict],
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> dict:
        def _score_wrapper(*, outputs: Union[str, dict], **kwargs):
            if code_extractor is None:
                normalized_outputs = _normalize_final_app_outputs_as_string(outputs)
                if code_extraction_strategy == "llm":
                    res = client.invoke(
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
                    normalized_outputs = res.content  # type: ignore
                elif code_extraction_strategy == "markdown_code_blocks":
                    normalized_outputs = _extract_code_from_markdown_code_blocks(
                        normalized_outputs
                    )
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
            raise ValueError("Either model or client must be provided")
        if client is None:
            client = init_chat_model(model)

    async def _wrapped_evaluator(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Union[str, dict],
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> SimpleAsyncEvaluator:
        async def _ascore_wrapper(*, outputs: Union[str, dict], **kwargs):
            if code_extractor is None:
                normalized_outputs = _normalize_final_app_outputs_as_string(outputs)
                if code_extraction_strategy == "llm":
                    res = await client.ainvoke(
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
                    normalized_outputs = res.content
                elif code_extraction_strategy == "markdown_code_blocks":
                    normalized_outputs = _extract_code_from_markdown_code_blocks(
                        normalized_outputs
                    )
                else:
                    # Nothing to do to extract code
                    pass
            else:
                normalized_outputs = code_extractor(outputs)
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
