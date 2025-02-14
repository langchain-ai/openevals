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
    metric: Optional[str] = None,
    system: Optional[str] = None,
    schema: Optional[dict] = None,
    judge: Optional[
        Union[
            ModelClient,
            LangChainLikeModel,
        ]
    ] = None,
    model: str = "openai:o3-mini",
    continuous: bool = False,
    use_reasoning: bool = True,
) -> Callable[..., Union[float, bool, dict]]:
    """
    Create a simple evaluator that uses an LLM to evaluate the quality of the outputs.
    """

    def get_score(
        *,
        outputs: Union[str, dict],
        inputs: Optional[Union[str, dict]] = None,
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> EvaluatorResult:
        if system is not None and not isinstance(prompt, str):
            raise ValueError(
                "`system` is only supported when `prompt` is a string template"
            )
            
        if isinstance(outputs, dict):
            outputs = json.dumps(outputs)
        if isinstance(reference_outputs, dict):
            reference_outputs = json.dumps(reference_outputs)
        if isinstance(inputs, dict):
            inputs = json.dumps(inputs)

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
        
        if system is not None:
            messages = [
                {"role": "system", "content": system},
                *messages,
            ]

        description = f"A numerical score measuring {metric}"
        json_schema = schema if schema is not None else {
                "type": "object",
                "additionalProperties": False,
        }
        # Make the output continuous or not
        if continuous and schema is None:
            description = f"A continuous score from 0 to 1 measuring {metric}"
            score_schema = {
                "type": "number",
                "description": description,
            }
        elif schema is None:
            description = f"A boolean of True or False. Only return True if the test case satisfies ALL the criteria for {metric}, i.e. the score = 1. If the score is less than 1 (by any amount), then return False. Only respond with True or False."
            score_schema = {
                "type": "boolean",
                "description": description,
            }
        
        # Add reasoning if passed
        if use_reasoning and schema is None:
            json_schema["properties"] = {
                "reasoning": {
                    "type": "string",
                    "description": "A human-readable explanation of the score",
                },
                "score": score_schema,
            }
            json_schema["required"] = ["score", "reasoning"]
        elif schema is None:
            json_schema["properties"] = {
                "score": score_schema,
            }
            json_schema["required"] = ["score"]

        nonlocal judge

        if judge is None:
            from langchain.chat_models import init_chat_model

            judge = init_chat_model(model=model)

        if isinstance(judge, LangChainLikeModel):
            response = judge.with_structured_output(
                {
                    "title": "score",
                    "description": description,
                    **json_schema,
                }
            ).invoke(messages)
            if schema is None:
                if use_reasoning:
                    return (response["score"], response["reasoning"])
                return response["score"]
            else:
                return response
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
            if schema is None:
                if use_reasoning:
                    return (parsed["score"], parsed["reasoning"])
                return parsed["score"]
            else:
                return parsed

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
    continuous: bool = False,
    use_reasoning: bool = True,
) -> SimpleEvaluator:
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
        inputs: dict,
        outputs: dict,
        reference_outputs: Optional[dict] = None,
        **kwargs,
    ) -> EvaluatorResult:
        return _run_evaluator(
            run_name=f"llm_as_{metric}_judge",
            scorer=scorer,
            feedback_key=metric,
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    return _wrapped_evaluator
