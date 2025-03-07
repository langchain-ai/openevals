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
