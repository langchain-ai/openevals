from evaluators.base import EvaluatorResult, SimpleEvaluator
from langsmith import testing as t
from langsmith.testing._internal import _TEST_CASE
from typing import (
    Callable,
    List,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)

import json


class ChatCompletionMessage(TypedDict):
    content: str
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


def create_llm_as_judge(
    *,
    prompt: str,
    client_or_scorer: Union[
        ModelClient, LangChainLikeModel, Callable[[List[ChatCompletionMessage]], float]
    ],
    model: Optional[str] = None,
    key: str = "quality",
) -> SimpleEvaluator:
    """
    Create a simple evaluator that uses an LLM to evaluate the quality of the outputs.
    """

    def wrapped_evaluator(
        *,
        inputs: dict,
        outputs: dict,
        reference_outputs: Optional[dict] = None,
        **kwargs,
    ) -> EvaluatorResult:
        formatted_prompt = prompt.format(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

        messages = [
            {"role": "user", "content": formatted_prompt},
        ]

        def get_score():
            json_schema = {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "A numerical score indicating the quality of the output",
                    }
                },
                "required": ["score"],
                "additionalProperties": False,
            }
            if isinstance(client_or_scorer, LangChainLikeModel):
                if model is not None:
                    raise ValueError("`model` is not allowed for LangChain clients")
                response = client_or_scorer.with_structured_output(
                    {
                        "title": "score",
                        "description": "A numerical score indicating the quality of the output",
                        **json_schema,
                    }
                ).invoke(messages)
                return response["score"]
            elif isinstance(client_or_scorer, ModelClient):
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
                response = client_or_scorer.chat.completions.create(**params)
                parsed = json.loads(response.choices[0].message.content)
                return parsed["score"]
            else:
                if model is not None:
                    raise ValueError("`model` is not allowed for arbitraryfunctions")
                return client_or_scorer(formatted_prompt)

        if _TEST_CASE.get():
            with t.trace_feedback():
                score = get_score()
                t.log_feedback(key=key, score=score)
        else:
            score = get_score()
        return EvaluatorResult(key=key, score=score)

    return wrapped_evaluator
