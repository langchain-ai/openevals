from evaluators.base import EvaluatorResult, SimpleEvaluator
from langsmith import testing as t
from langsmith import traceable
from langsmith.testing._internal import _TEST_CASE
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


def create_llm_as_judge(
    *,
    prompt: str | RunnableLike | Callable[..., list[ChatCompletionMessage]],
    key: str = "quality",
    model: Optional[str] = None,
    client: Optional[ModelClient] = None,
    continuous: bool = False,
    use_reasoning: bool = True,
) -> SimpleEvaluator:
    """
    Create a simple evaluator that uses an LLM to evaluate the quality of the outputs.
    """

    def wrapped_evaluator(
        *,
        inputs: dict,
        outputs: dict,
        reference_outputs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
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

        def get_score():
            json_schema = {
                "type": "object",
                "additionalProperties": False,
            }
            # Make the output continuous or not
            if continuous:
                description = f"A continuous score from 0 to 1 indicating how well the test case satisfies the criteria"
                score_schema = {
                    "type": "number",
                    "description": description,
                }
            else:
                description = f"A boolean of True or False. Only return True if the test case satisfies ALL the criteria, i.e. the score = 1. If the score is less than 1 (by any amount), then return False. Remember this is a BOOLEAN. Only respond with True or False."
                score_schema = {
                    "type": "boolean",
                    "description": description,
                }
            
            # Add reasoning if passed
            if use_reasoning:
                json_schema["properties"] = {
                    "reasoning": {
                        "type": "string",
                        "description": "A human-readable explanation of the score",
                    },
                    "score": score_schema,
                }
                json_schema["required"] = ["score", "reasoning"]
            else:
                json_schema["properties"] = {
                    "score": score_schema,
                }
                json_schema["required"] = ["score"]

            nonlocal client

            if client is None:
                if model is None:
                    model_to_use = "openai:gpt-4o"
                else:
                    model_to_use = model
                from langchain.chat_models import init_chat_model

                try:
                    judge = init_chat_model(model=model_to_use, **(model_kwargs or {}))
                except ValueError:
                    raise ValueError(
                        f"Could not find model: {model_to_use}."
                    )
                json_schema['title'] = "evaluator_score"
                json_schema['description'] = "The score for the evaluation criteria"
                judge = judge.with_structured_output(
                    json_schema
                )
                response = judge.invoke(messages)
                return response["score"], response.get("reasoning")
                

            elif isinstance(client, ModelClient):
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
                    **(model_kwargs or {}),
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
                return parsed["score"], parsed.get("reasoning")

        if _TEST_CASE.get():
            with t.trace_feedback():
                score, reasoning = get_score()
                t.log_feedback(key=key, score=score)
        else:
            score, reasoning = get_score()
        return EvaluatorResult(key=key, score=score, comment=reasoning)

    return wrapped_evaluator
