from __future__ import annotations
from evaluators.llm import (
    _create_llm_as_judge_scorer,
    _create_async_llm_as_judge_scorer,
    ChatCompletionMessage,
    RunnableLike,
    ModelClient,
    LangChainLikeModel,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
    Callable,
    Optional,
    Union,
)
from evaluators.types import EvaluatorResult, FewShotExample
from evaluators.utils import (
    _chat_completion_messages_to_string,
    _run_evaluator,
    _arun_evaluator,
    _normalize_to_openai_messages_list,
)
from typing import TYPE_CHECKING

DEFAULT_PROMPT = """Grade the following agent trajectory:

<trajectory>
{outputs}
</trajectory>
{inputs}
{reference_outputs}
{rubric}
"""

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def _format_inputs(
    inputs: Optional[dict],
    outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
    reference_outputs: Optional[
        Union[list[ChatCompletionMessage], list[BaseMessage], dict]
    ],
    rubric: Optional[str],
) -> tuple[str, str, str, str]:
    outputs = _normalize_to_openai_messages_list(outputs)
    reference_outputs = _normalize_to_openai_messages_list(reference_outputs)
    if reference_outputs:
        formatted_reference_outputs = f"\nUse the following trajectory as an example reference when grading:\n<reference_trajectory>\n{_chat_completion_messages_to_string(reference_outputs)}\n</reference_trajectory>\n"
    else:
        formatted_reference_outputs = ""
    if inputs:
        formatted_inputs = f"\nThe agent generated the trajectory from the following input:\n<input>\n{inputs}\n</input>\n"
    else:
        formatted_inputs = ""
    if isinstance(outputs, dict):
        formatted_outputs = outputs
    else:
        formatted_outputs = _chat_completion_messages_to_string(outputs)
    if rubric:
        formatted_rubric = f"\nGrade the agent trajectory along the following rubric:\n<rubric>\n{rubric}\n</rubric>\n"
    else:
        formatted_rubric = ""
    return (
        formatted_outputs,
        formatted_reference_outputs,
        formatted_inputs,
        formatted_rubric,
    )


def create_trajectory_llm_as_judge(
    *,
    prompt: str
    | RunnableLike
    | Callable[..., list[ChatCompletionMessage]] = DEFAULT_PROMPT,
    model: str = "openai:o3-mini",
    feedback_key: str = "trajectory_accuracy",
    judge: Optional[
        Union[
            ModelClient,
            LangChainLikeModel,
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
            that returns a list of chat messages. Note that the default prompt allows a rubric
            in addition to the typical "inputs", "outputs", and "reference_outputs" parameters.
        feedback_key: Key used to store the evaluation result, defaults to "score".
        judge: The LLM used for evaluation. Can be an OpenAI client),
            or a LangChainLikeModel. If an OpenAI client, must specify "model" as well.
            If omitted, "model" will be used to instantiate a LangChain model instance
            by model string.
        model: Model identifier to use. Defaults to "openai:o3-mini". If "judge" is an OpenAI client,
            this argument should be a model name directly. If "judge" is omitted, must be a valid
            LangChain model identifier. See `init_chat_model` docs for more details:
            https://python.langchain.com/docs/how_to/chat_models_universal_init/.
        system: Optional system message to prepend to the prompt.
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
        inputs: Optional[dict] = None,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        reference_outputs: Optional[
            Union[list[ChatCompletionMessage], list[BaseMessage], dict]
        ] = None,
        rubric: Optional[str] = None,
        **kwargs,
    ) -> EvaluatorResult:
        (
            formatted_outputs,
            formatted_reference_outputs,
            formatted_inputs,
            formatted_rubric,
        ) = _format_inputs(inputs, outputs, reference_outputs, rubric)
        return _run_evaluator(
            run_name=f"llm_as_{feedback_key}_judge",
            scorer=scorer,
            feedback_key=feedback_key,
            outputs=formatted_outputs,
            reference_outputs=formatted_reference_outputs,
            inputs=formatted_inputs,
            rubric=formatted_rubric,
            **kwargs,
        )

    return _wrapped_evaluator


def create_async_trajectory_llm_as_judge(
    *,
    prompt: str
    | RunnableLike
    | Callable[..., list[ChatCompletionMessage]] = DEFAULT_PROMPT,
    model: str = "openai:o3-mini",
    feedback_key: str = "trajectory_accuracy",
    judge: Optional[
        Union[
            ModelClient,
            LangChainLikeModel,
        ]
    ] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleAsyncEvaluator:
    """Creates an evaluator that uses an LLM to judge agent trajectories.

    Args:
        prompt: The evaluation prompt, can be a string template, LangChain prompt template, or callable
            that returns a list of chat messages. Note that the default prompt allows a rubric
            in addition to the typical "inputs", "outputs", and "reference_outputs" parameters.
        feedback_key: Key used to store the evaluation result, defaults to "score".
        judge: The LLM used for evaluation. Can be an OpenAI client),
            or a LangChainLikeModel. If an OpenAI client, must specify "model" as well.
            If omitted, "model" will be used to instantiate a LangChain model instance
            by model string.
        model: Model identifier to use. Defaults to "openai:o3-mini". If "judge" is an OpenAI client,
            this argument should be a model name directly. If "judge" is omitted, must be a valid
            LangChain model identifier. See `init_chat_model` docs for more details:
            https://python.langchain.com/docs/how_to/chat_models_universal_init/.
        system: Optional system message to prepend to the prompt.
        continuous: If True, score will be a float between 0 and 1. If False, score will be boolean. Defaults to False.
        choices: Optional list of specific float values the score must be chosen from.
        use_reasoning: If True, includes explanation for the score in the output. Defaults to True.
        few_shot_examples: Optional list of example evaluations to append to the prompt.

    Returns:
        SimpleAsyncEvaluator: A function that evaluates agent trajectories using the configured LLM judge.
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
        inputs: Optional[dict] = None,
        outputs: Union[list[ChatCompletionMessage], list[BaseMessage], dict],
        reference_outputs: Optional[
            Union[list[ChatCompletionMessage], list[BaseMessage], dict]
        ] = None,
        rubric: Optional[str] = None,
        **kwargs,
    ) -> EvaluatorResult:
        (
            formatted_outputs,
            formatted_reference_outputs,
            formatted_inputs,
            formatted_rubric,
        ) = _format_inputs(inputs, outputs, reference_outputs, rubric)
        return await _arun_evaluator(
            run_name=f"llm_as_{feedback_key}_judge",
            scorer=scorer,
            feedback_key=feedback_key,
            outputs=formatted_outputs,
            reference_outputs=formatted_reference_outputs,
            inputs=formatted_inputs,
            rubric=formatted_rubric,
            **kwargs,
        )

    return _wrapped_evaluator
