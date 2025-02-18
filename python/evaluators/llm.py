from evaluators.utils import _run_evaluator, _arun_evaluator
from evaluators.types import (
    EvaluatorResult,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
    RunnableLike,
    ModelClient,
    ChatCompletionMessage,
    FewShotExample,
)

from langchain.chat_models import init_chat_model
from langsmith import traceable

from typing import (
    Callable,
    Optional,
    Union,
    TYPE_CHECKING,
)

import json

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def _append_few_shot_examples(
    *,
    messages: list[ChatCompletionMessage],
    few_shot_examples: list[FewShotExample],
) -> list[ChatCompletionMessage]:
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
            + (f"\n<score>{example['score']}</score>" if "score" in example else "")
            + "\n</example>"
            for example in few_shot_examples
        ]
    )
    return messages


def _construct_output_schema(
    *,
    schema: Optional[dict] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
) -> tuple[dict, str]:
    json_schema = (
        schema
        if schema is not None
        else {
            "type": "object",
            "additionalProperties": False,
            "strict": True,
        }
    )
    # Set the description for the score schema
    if choices:
        description = "A number that represents the degree to which the criteria in the prompt are met."
        score_schema = {
            "type": "number",
            "description": description,
            "enum": choices,
        }
    elif continuous:
        description = "A number that represents the degree to which the criteria in the prompt are met, from 0.0 to 1.0. 1.0 means the criteria are met perfectly. 0.0 means none of the criteria are met."
        score_schema = {
            "type": "number",
            "description": description,
        }
    else:
        description = "A score that is true if criteria in the prompt are met, and false otherwise."
        score_schema = {
            "type": "boolean",
            "description": description,
        }

    # Add reasoning if passed
    if schema is None:
        if use_reasoning:
            json_schema["properties"] = {
                "reasoning": {
                    "type": "string",
                    "description": "A human-readable explanation of the score. You MUST end the reasoning with a sentence that says: Thus, the score should be: SCORE_YOU_ASSIGN.",
                },
                "score": score_schema,
            }
            json_schema["required"] = ["reasoning", "score"]
        else:
            json_schema["properties"] = {
                "score": score_schema,
            }
            json_schema["required"] = ["score"]

    return (json_schema, description)


def _create_llm_as_judge_scorer(
    *,
    prompt: str | RunnableLike | Callable[..., list[ChatCompletionMessage]],
    system: Optional[str] = None,
    schema: Optional[dict] = None,
    judge: Optional[
        Union[
            ModelClient,
            BaseChatModel,
        ]
    ] = None,
    model: str = "openai:o3-mini",
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleEvaluator:
    def get_score(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Union[str, dict],
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ):
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

        # Add few shot examples to the prompt
        if few_shot_examples:
            messages = _append_few_shot_examples(
                messages=messages,
                few_shot_examples=few_shot_examples,
            )

        (json_schema, description) = _construct_output_schema(
            schema=schema,
            continuous=continuous,
            choices=choices,
            use_reasoning=use_reasoning,
        )

        nonlocal judge

        if judge is None:
            judge = init_chat_model(model=model)

        if isinstance(judge, BaseChatModel):
            response = judge.with_structured_output(
                {
                    "title": "score",
                    "description": description,
                    **json_schema,
                }
            ).invoke(messages)
            if schema is None:
                if use_reasoning:
                    return (response["score"], response["reasoning"])  # type: ignore
                return response["score"]  # type: ignore
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
            parsed = json.loads(response.choices[0].message.content)  # type: ignore
            if schema is None:
                if use_reasoning:
                    return (parsed["score"], parsed["reasoning"])  # type: ignore
                return parsed["score"]  # type: ignore
            else:
                return parsed
        else:
            raise ValueError("`judge` must be a ModelClient or BaseChatModel")

    return get_score


def _create_async_llm_as_judge_scorer(
    *,
    prompt: str | RunnableLike | Callable[..., list[ChatCompletionMessage]],
    system: Optional[str] = None,
    schema: Optional[dict] = None,
    judge: Optional[
        Union[
            ModelClient,
            BaseChatModel,
        ]
    ] = None,
    model: str = "openai:o3-mini",
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleAsyncEvaluator:
    async def aget_score(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Union[str, dict],
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ):
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
            formatted_prompt = await prompt.ainvoke(
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
            messages = _append_few_shot_examples(
                messages=messages,
                few_shot_examples=few_shot_examples,
            )

        (json_schema, description) = _construct_output_schema(
            schema=schema,
            continuous=continuous,
            choices=choices,
            use_reasoning=use_reasoning,
        )

        nonlocal judge

        if judge is None:
            judge = init_chat_model(model=model)

        if isinstance(judge, BaseChatModel):
            response = await judge.with_structured_output(
                {
                    "title": "score",
                    "description": description,
                    **json_schema,
                }
            ).ainvoke(messages)
            if schema is None:
                if use_reasoning:
                    return (response["score"], response["reasoning"])  # type: ignore
                return response["score"]  # type: ignore
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
            async def ainvoke_llm(**params):
                return await judge.chat.completions.create(**params)

            response = await ainvoke_llm(**params)
            parsed = json.loads(response.choices[0].message.content)  # type: ignore
            if schema is None:
                if use_reasoning:
                    return (parsed["score"], parsed["reasoning"])  # type: ignore
                return parsed["score"]  # type: ignore
            else:
                return parsed
        else:
            raise ValueError("`judge` must be a ModelClient or BaseChatModel")

    return aget_score


def create_llm_as_judge(
    *,
    prompt: str | RunnableLike | Callable[..., list[ChatCompletionMessage]],
    feedback_key: str = "score",
    judge: Optional[Union[ModelClient, BaseChatModel]] = None,
    model: str = "openai:o3-mini",
    system: Optional[str] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleEvaluator:
    """Create an evaluator that uses an LLM to assess output quality based on specified criteria.
    Handles prompt formatting, LLM invocation, and structured output parsing.

    Args:
        prompt: The evaluation prompt, can be a string template, LangChain prompt template, or callable
            that returns a list of chat messages.
        feedback_key: Key used to store the evaluation result, defaults to "score".
        judge: The LLM used for evaluation. Can be an OpenAI client),
            or a BaseChatModel. If an OpenAI client, must specify "model" as well.
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
        A function that takes inputs, outputs, reference_outputs, and other kwargs, formats them into
        a prompt, invokes the judge, and returns an evaluation result.

    Example:
        ```python
        from openevals.evaluators.llm import create_llm_as_judge

        evaluator = create_llm_as_judge(
            prompt="Rate the quality of this response from 0 to 1: {outputs}",
            continuous=True
        )
        result = evaluator(
            inputs={"question": "What color is the sky?"},
            outputs={"response": "Blue"},
        )
        ```
    """
    scorer = _create_llm_as_judge_scorer(
        prompt=prompt,
        judge=judge,
        system=system,
        model=model,
        continuous=continuous,
        choices=choices,
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
        run_name = (
            "llm_as_judge"
            if feedback_key == "score"
            else f"llm_as_{feedback_key}_judge"
        )
        res = _run_evaluator(
            run_name=run_name,
            scorer=scorer,
            feedback_key=feedback_key,
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )
        if isinstance(res, list):
            return res[0]
        return res  # type: ignore

    return _wrapped_evaluator  # type: ignore


def create_async_llm_as_judge(
    *,
    prompt: str | RunnableLike | Callable[..., list[ChatCompletionMessage]],
    feedback_key: str = "score",
    judge: Optional[Union[ModelClient, BaseChatModel]] = None,
    model: str = "openai:o3-mini",
    system: Optional[str] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> SimpleAsyncEvaluator:
    """Create an evaluator that uses an LLM to assess output quality based on specified criteria.
    Handles prompt formatting, LLM invocation, and structured output parsing.

    Args:
        prompt: The evaluation prompt, can be a string template, LangChain prompt template, or callable
            that returns a list of chat messages.
        feedback_key: Key used to store the evaluation result, defaults to "score".
        judge: The LLM used for evaluation. Can be an OpenAI client),
            or a BaseChatModel. If an OpenAI client, must specify "model" as well.
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
        A function that takes inputs, outputs, reference_outputs, and other kwargs, formats them into
        a prompt, invokes the judge, and returns an evaluation result.

    Example:
        ```python
        from openevals.evaluators.llm import create_async_llm_as_judge

        evaluator = create_async_llm_as_judge(
            prompt="Rate the quality of this response from 0 to 1: {outputs}",
            continuous=True
        )
        result = await evaluator(
            inputs={"question": "What color is the sky?"},
            outputs={"response": "Blue"},
        )
        ```
    """
    scorer = _create_async_llm_as_judge_scorer(
        prompt=prompt,
        judge=judge,
        system=system,
        model=model,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
    )

    async def _wrapped_evaluator(
        *,
        inputs: dict,
        outputs: dict,
        reference_outputs: Optional[dict] = None,
        **kwargs,
    ) -> EvaluatorResult:
        run_name = (
            "llm_as_judge"
            if feedback_key == "score"
            else f"llm_as_{feedback_key}_judge"
        )
        res = await _arun_evaluator(
            run_name=run_name,
            scorer=scorer,
            feedback_key=feedback_key,
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )
        if isinstance(res, list):
            return res[0]
        return res

    return _wrapped_evaluator  # type: ignore
