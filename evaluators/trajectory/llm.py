from __future__ import annotations
from evaluators.llm import (
    _create_llm_as_judge_scorer,
    ChatCompletionMessage,
    RunnableLike,
    ModelClient,
    LangChainLikeModel,
    Callable,
    Optional,
    Union,
)
from evaluators.types import EvaluatorResult, FewShotExample
from evaluators.utils import (
    _chat_completion_messages_to_string,
    _run_evaluator,
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
):
    scorer = _create_llm_as_judge_scorer(
        prompt=prompt,
        judge=judge,
        model=model,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )

    if TYPE_CHECKING:
        from langchain_core.messages import BaseMessage

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
            formatted_rubric = f"\nGrade the following agent trajectory along the following rubric:\n<rubric>\n{rubric}\n</rubric>\n"
        else:
            formatted_rubric = ""
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
