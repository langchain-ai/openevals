from __future__ import annotations

from openevals.llm import (
    _create_llm_as_judge_scorer,
    _create_async_llm_as_judge_scorer,
)
from openevals.types import (
    ChatCompletionMessage,
    ModelClient,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
    FewShotExample,
)
from openevals.utils import (
    _run_evaluator,
    _arun_evaluator,
    _chat_completion_messages_to_string,
)
from openevals.trajectory.utils import _normalize_to_openai_messages_list
from openevals.prompts.trajectory import (
    TRAJECTORY_ACCURACY_PROMPT,
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from typing import Callable, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

__all__ = [
    "TRAJECTORY_ACCURACY_PROMPT",
    "TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE",
    "create_trajectory_llm_as_judge",
    "create_async_trajectory_llm_as_judge",
]


def _format_inputs(
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Optional[
        Union[list[ChatCompletionMessage], list[BaseMessage], dict]
    ],
) -> tuple[str, str]:
    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)
    if reference_outputs:
        formatted_reference_outputs = _chat_completion_messages_to_string(
            reference_outputs
        )
    else:
        formatted_reference_outputs = ""
    formatted_outputs = _chat_completion_messages_to_string(outputs)
    return (
        formatted_outputs,
        formatted_reference_outputs,
    )


def create_trajectory_llm_as_judge(
    *,
    prompt: str
    | Runnable
    | Callable[
        ..., list[ChatCompletionMessage]
    ] = TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    model: Optional[str] = None,
    feedback_key: str = "trajectory_accuracy",
    judge: Optional[
        Union[
            ModelClient,
            BaseChatModel,
        ]
    ] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleEvaluator:
    """Creates an evaluator that uses an LLM to judge agent trajectories.

    Args:
        prompt: The evaluation prompt, can be a string template, LangChain prompt template, or callable
            that returns a list of chat messages.
        feedback_key: Key used to store the evaluation result, defaults to "trajectory_accuracy".
        judge: The LLM used for evaluation. Can be an OpenAI client or a LangChain chat model.
        model: Model identifier to use.
        continuous: If True, score will be a float between 0 and 1. If False, score will be boolean. Defaults to False.
        choices: Optional list of specific float values the score must be chosen from.
        use_reasoning: If True, includes explanation for the score in the output. Defaults to True.
        few_shot_examples: Optional list of example evaluations to append to the prompt.

    Returns:
        SimpleEvaluator: A function that evaluates agent trajectories using the configured LLM judge.
    """
    scorer = _create_llm_as_judge_scorer(
        prompt=prompt,
        judge=judge,
        model=model,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )

    def _wrapped_evaluator(
        *,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        reference_outputs: Optional[
            Union[list[ChatCompletionMessage], list[BaseMessage], dict]
        ] = None,
        **kwargs,
    ):
        (
            formatted_outputs,
            formatted_reference_outputs,
        ) = _format_inputs(outputs, reference_outputs)
        return _run_evaluator(
            run_name=f"llm_as_{feedback_key}_judge",
            scorer=scorer,
            feedback_key=feedback_key,
            outputs=formatted_outputs,
            reference_outputs=formatted_reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator


def create_async_trajectory_llm_as_judge(
    *,
    prompt: str
    | Runnable
    | Callable[
        ..., list[ChatCompletionMessage]
    ] = TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    model: Optional[str] = None,
    feedback_key: str = "trajectory_accuracy",
    judge: Optional[
        Union[
            ModelClient,
            BaseChatModel,
        ]
    ] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleAsyncEvaluator:
    """Creates an async evaluator that uses an LLM to judge agent trajectories.

    Args:
        prompt: The evaluation prompt, can be a string template, LangChain prompt template, or callable
            that returns a list of chat messages.
        feedback_key: Key used to store the evaluation result, defaults to "trajectory_accuracy".
        judge: The LLM used for evaluation. Can be an OpenAI client or a LangChain chat model.
        model: Model identifier to use.
        continuous: If True, score will be a float between 0 and 1. If False, score will be boolean. Defaults to False.
        choices: Optional list of specific float values the score must be chosen from.
        use_reasoning: If True, includes explanation for the score in the output. Defaults to True.
        few_shot_examples: Optional list of example evaluations to append to the prompt.

    Returns:
        SimpleAsyncEvaluator: An async function that evaluates agent trajectories using the configured LLM judge.
    """
    scorer = _create_async_llm_as_judge_scorer(
        prompt=prompt,
        judge=judge,
        model=model,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )

    async def _wrapped_evaluator(
        *,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        reference_outputs: Optional[
            Union[list[ChatCompletionMessage], list[BaseMessage], dict]
        ] = None,
        **kwargs,
    ):
        (
            formatted_outputs,
            formatted_reference_outputs,
        ) = _format_inputs(outputs, reference_outputs)
        return await _arun_evaluator(
            run_name=f"llm_as_{feedback_key}_judge",
            scorer=scorer,
            feedback_key=feedback_key,
            outputs=formatted_outputs,
            reference_outputs=formatted_reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator
