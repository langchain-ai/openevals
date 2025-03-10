from openevals.code.base import (
    _create_base_code_evaluator,
    _create_async_base_code_evaluator,
)
from openevals.llm import _create_llm_as_judge_scorer, _create_async_llm_as_judge_scorer
from openevals.prompts import (
    CODE_CORRECTNESS_PROMPT,
    CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS,
)

from typing import Callable, Optional, Union, Literal, Any

from langchain_core.language_models.chat_models import BaseChatModel

from openevals.types import (
    SimpleEvaluator,
    SimpleAsyncEvaluator,
    RunnableLike,
    ModelClient,
    ChatCompletionMessage,
    FewShotExample,
)

__all__ = [
    "create_code_llm_as_judge",
    "CODE_CORRECTNESS_PROMPT",
    "CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS",
]


def create_code_llm_as_judge(
    *,
    prompt: str | RunnableLike | Callable[..., list[ChatCompletionMessage]],
    feedback_key: str = "code_correctness",
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    judge: Optional[Union[ModelClient, BaseChatModel]] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleEvaluator:
    """Creates an evaluator that uses an LLM to judge code correctness.

    This function creates an evaluator that extracts code from a response and uses
    an LLM to judge its correctness based on the provided prompt and criteria.

    Takes all the same arguments as the `create_llm_as_judge` function.

    Args:
        prompt: The prompt to use for evaluation. Can be a string, a runnable, or a
            callable that returns a list of chat messages.
        feedback_key: The key to use for storing feedback in the evaluation results.
            Defaults to "code_correctness".
        code_extraction_strategy: Strategy for extracting code from the response.
            Options are "none" (use raw response), "llm" (use LLM to extract code),
            or "markdown_code_blocks" (extract code from markdown blocks).
            Defaults to "none".
        code_extractor: Optional custom function to extract code from the response.
            If provided, overrides the code_extraction_strategy.
        judge: The model client or LangChain chat model to use as the judge.
            If not provided, will use the model specified by the model parameter.
        model: The name of the model to use if judge is not provided.
        system: Optional system message to include in the prompt to the judge.
        continuous: Whether to return a continuous score. If False, returns a
            categorical score based on choices. Defaults to False.
        choices: Optional list of possible score values when continuous is False.
        use_reasoning: Whether to include reasoning in the evaluation output.
            Defaults to True.
        few_shot_examples: Optional list of few-shot examples to include in the prompt.

    Returns:
        A SimpleEvaluator that evaluates code correctness using an LLM as judge.
    """
    scorer = _create_llm_as_judge_scorer(
        prompt=prompt,
        system=system,
        model=model,
        judge=judge,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )
    return _create_base_code_evaluator(
        model=model,
        client=judge,
        run_name="code_llm_as_judge",
        feedback_key=feedback_key or "code_correctness",
        scorer=scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
    )


def create_async_code_llm_as_judge(
    *,
    prompt: str | RunnableLike | Callable[..., list[ChatCompletionMessage]],
    feedback_key: str = "code_correctness",
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    judge: Optional[Union[ModelClient, BaseChatModel]] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleAsyncEvaluator:
    """Creates an evaluator that uses an LLM to judge code correctness.

    This function creates an asynchronous evaluator that extracts code from a response and uses
    an LLM to judge its correctness based on the provided prompt and criteria.

    Takes all the same arguments as the `create_async_llm_as_judge` function.

    Args:
        prompt: The prompt to use for evaluation. Can be a string, a runnable, or a
            callable that returns a list of chat messages.
        feedback_key: The key to use for storing feedback in the evaluation results.
            Defaults to "code_correctness".
        code_extraction_strategy: Strategy for extracting code from the response.
            Options are "none" (use raw response), "llm" (use LLM to extract code),
            or "markdown_code_blocks" (extract code from markdown blocks).
            Defaults to "none".
        code_extractor: Optional custom function to extract code from the response.
            If provided, overrides the code_extraction_strategy.
        judge: The model client or LangChain chat model to use as the judge.
            If not provided, will use the model specified by the model parameter.
        model: The name of the model to use if judge is not provided.
        system: Optional system message to include in the prompt to the judge.
        continuous: Whether to return a continuous score. If False, returns a
            categorical score based on choices. Defaults to False.
        choices: Optional list of possible score values when continuous is False.
        use_reasoning: Whether to include reasoning in the evaluation output.
            Defaults to True.
        few_shot_examples: Optional list of few-shot examples to include in the prompt.

    Returns:
        A SimpleAsyncEvaluator that evaluates code correctness using an LLM as judge.
    """
    scorer = _create_async_llm_as_judge_scorer(
        prompt=prompt,
        system=system,
        model=model,
        judge=judge,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )
    return _create_async_base_code_evaluator(
        model=model,
        client=judge,
        run_name="code_llm_as_judge",
        feedback_key=feedback_key or "code_correctness",
        scorer=scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
    )
