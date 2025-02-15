from evaluators.utils import _run_evaluator
from evaluators.types import (
    EvaluatorResult,
    SimpleEvaluator,
    RunnableLike,
    LangChainLikeModel,
    ModelClient,
    ChatCompletionMessage,
    FewShotExample,
)

from langsmith import traceable

from typing import (
    Callable,
    Optional,
    Union,
)

import json


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
    threshold: Optional[float] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> Callable[..., Union[float, bool]]:
    """
    Create a simple evaluator that uses an LLM to evaluate the quality of the outputs.
    """

    def get_score(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Union[str, dict],
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

        if threshold is not None and (threshold < 0 or threshold > 1):
            raise ValueError("Threshold must be between 0 and 1")


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
        
        # Add few shot examples to the prompt
        if few_shot_examples:
            # Find the last user message to append examples to
            last_user_message_idx = None
            for i, msg in enumerate(messages[::-1]):
                if msg.get("role") == "user":
                    last_user_message_idx = len(messages) - 1 - i
                    break

            if last_user_message_idx is None:
                raise ValueError(
                    "Appending few-shot examples requires a user message in the provided prompt"
                )

            messages[last_user_message_idx]["content"] += "\n\n" + "\n".join(
                [
                    f"<example>\n<input>{example['inputs']}</input>\n<output>{example['outputs']}</output>"
                    + (
                        f"\n<reasoning>{example['reasoning']}</reasoning>"
                        if "reasoning" in example
                        else ""
                    )
                    + (
                        f"\n<score>{example['score']}</score>"
                        if "score" in example
                        else ""
                    )
                    + "\n</example>"
                    for example in few_shot_examples
                ]
            )

        description = f"A numerical score measuring {metric}"
        json_schema = schema if schema is not None else {
                "type": "object",
                "additionalProperties": False,
        }
        
        # Set the description for the score schema
        description = f"The proportion of the criteria that are met for {metric}, a number from 0.0 to 1.0. 1.0 means all the criteria are met. 0.0 means none of the criteria are met."
        score_schema = {
            "type": "number",
            "description": description,
        }

        # Add reasoning if passed
        if use_reasoning and schema is None:
            json_schema["properties"] = {
                "reasoning": {
                    "type": "string",
                    "description": "A human-readable explanation of the score. You MUST end the reasoning with a sentence that says: Thus, the score should be: SCORE_YOU_ASSIGN. Where SCORE_YOU_ASSIGN is a single number between 0.0 and 1.0",
                },
                "score": score_schema,
            }
            json_schema["required"] = ["reasoning", "score"]
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
    threshold: Optional[float] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleEvaluator:
    scorer = _create_llm_as_judge_scorer(
        prompt=prompt,
        metric=metric,
        judge=judge,
        model=model,
        threshold=threshold,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
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
