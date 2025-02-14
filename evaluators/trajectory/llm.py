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
from evaluators.types import EvaluatorResult
from evaluators.utils import _chat_completion_messages_to_string, _run_evaluator

DEFAULT_PROMPT = """Grade the following agent trajectory:

<trajectory>
{outputs}
</trajectory>
{formatted_inputs_if_provided}
{formatted_reference_outputs_if_provided}
{formatted_rubric_if_provided}
"""


def create_trajectory_llm_as_judge(
    *,
    prompt: str
    | RunnableLike
    | Callable[..., list[ChatCompletionMessage]] = DEFAULT_PROMPT,
    metric: str = "trajectory_accuracy",
    judge: Optional[
        Union[
            ModelClient,
            LangChainLikeModel,
        ]
    ] = None,
    model: Optional[str] = None,
    continuous: bool = False,
    use_reasoning: bool = True,
):
    scorer = _create_llm_as_judge_scorer(
        prompt=prompt,
        metric=metric,
        judge=judge,
        model=model,
        continuous=continuous,
        use_reasoning=use_reasoning,
    )

    def _wrapped_evaluator(
        *,
        inputs: Optional[dict] = None,
        outputs: list[ChatCompletionMessage] | dict,
        reference_outputs: Optional[dict] = None,
        rubric: Optional[str] = None,
        **kwargs,
    ) -> EvaluatorResult:
        if reference_outputs:
            formatted_reference_outputs = f"\nUse the following trajectory as an example reference when grading:\n<reference_trajectory>\n{reference_outputs}\n</reference_trajectory>\n"
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
            run_name=f"trajectory_{metric}_llm_as_judge",
            scorer=scorer,
            feedback_key=metric,
            outputs=formatted_outputs,
            formatted_reference_outputs_if_provided=formatted_reference_outputs,
            formatted_inputs_if_provided=formatted_inputs,
            formatted_rubric_if_provided=formatted_rubric,
            **kwargs,
        )

    return _wrapped_evaluator
