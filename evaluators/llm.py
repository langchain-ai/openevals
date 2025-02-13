from evaluators.utils import _run_evaluator
from evaluators.types import EvaluatorResult, SimpleEvaluator

from langsmith import traceable

from typing import (
    Callable,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)

import json


class ChatCompletionMessage(TypedDict):
    content: list[Union[str, dict]]
    role: str


class ChatCompletion(TypedDict):
    choices: list[dict]


@runtime_checkable
class ChatCompletionsClient(Protocol):
    def create(self, **kwargs) -> ChatCompletion: ...


@runtime_checkable
class ModelClient(Protocol):
    @property
    def chat(self) -> type[ChatCompletionsClient]: ...


@runtime_checkable
class RunnableLike(Protocol):
    def invoke(self, **kwargs) -> ChatCompletion: ...


@runtime_checkable
class LangChainLikeModel(Protocol):
    def with_structured_output(self, **kwargs) -> RunnableLike: ...


def _create_llm_as_judge_scorer(
    *,
    prompt: str | RunnableLike | Callable[..., list[ChatCompletionMessage]],
    metric: str,
    judge: Optional[
        Union[
            ModelClient,
            LangChainLikeModel,
            Callable[[list[ChatCompletionMessage]], float],
        ]
    ] = None,
    model: Optional[str] = None,
) -> Callable[..., Union[float, bool]]:
    """
    Create a simple evaluator that uses an LLM to evaluate the quality of the outputs.
    """

    def get_score(
        *,
        inputs: dict,
        outputs: dict,
        reference_outputs: Optional[dict] = None,
        **kwargs,
    ) -> EvaluatorResult:
        if isinstance(prompt, RunnableLike):
            formatted_prompt = prompt.invoke(
                inputs=inputs,
                outputs=outputs,
                reference_outputs=reference_outputs,
                **kwargs,
            )
            messages = formatted_prompt.messages
        elif isinstance(prompt, str):
            formatted_prompt = prompt.format(
                inputs=inputs,
                outputs=outputs,
                reference_outputs=reference_outputs,
                **kwargs,
            )
            messages = [
                {"role": "user", "content": formatted_prompt},
            ]
        else:
            messages = prompt(
                inputs=inputs,
                outputs=outputs,
                reference_outputs=reference_outputs,
                **kwargs,
            )
        description = f"A numerical score measuring {metric}"
        json_schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": description,
                }
            },
            "required": ["score"],
            "additionalProperties": False,
        }

        nonlocal judge

        if judge is None:
            from langchain.chat_models import init_chat_model

            judge = init_chat_model(model=model or "openai:gpt-4o-mini")

        if isinstance(judge, LangChainLikeModel):
            response = judge.with_structured_output(
                {
                    "title": "score",
                    "description": description,
                    **json_schema,
                }
            ).invoke(messages)
            return response["score"]
        elif isinstance(judge, ModelClient):
            if model is None:
                raise ValueError("`model` is required for non-LangChain clients")
            params = {
                "messages": messages,
                "model": model,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "score",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            }

            @traceable(
                run_type="llm",
                metadata={
                    "ls_provider": "openai",
                    "ls_model_name": model,
                    "ls_model_type": "chat",
                },
            )
            def invoke_llm(**params):
                return judge.chat.completions.create(**params)

            response = invoke_llm(**params)
            parsed = json.loads(response.choices[0].message.content)
            return parsed["score"]
        else:
            if model is not None:
                raise ValueError(
                    "`model` is not allowed when passing a raw function as the judge"
                )
            return judge(formatted_prompt)

    return get_score


def create_llm_as_judge(
    *,
    prompt: str | RunnableLike | Callable[..., list[ChatCompletionMessage]],
    metric: str = "quality",
    judge: Optional[
        Union[
            ModelClient,
            LangChainLikeModel,
            Callable[[list[ChatCompletionMessage]], float],
        ]
    ] = None,
    model: Optional[str] = None,
) -> SimpleEvaluator:
    evaluator = _create_llm_as_judge_scorer(
        prompt=prompt, metric=metric, judge=judge, model=model
    )

    def _wrapped_evaluator(
        *,
        inputs: dict,
        outputs: dict,
        reference_outputs: Optional[dict] = None,
        **kwargs,
    ) -> EvaluatorResult:
        return _run_evaluator(
            run_name=f"llm_as_{metric}_judge",
            evaluator_fn=evaluator,
            feedback_key=metric,
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator
