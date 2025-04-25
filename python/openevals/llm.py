from __future__ import annotations
from openevals.utils import (
    _run_evaluator_untyped,
    _arun_evaluator_untyped,
    _convert_to_openai_message,
    _normalize_to_openai_messages_list,
)
from openevals.types import (
    EvaluatorResult,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
    ModelClient,
    ChatCompletionMessage,
    FewShotExample,
    ScoreType,
)

from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel, BaseMessage
from langchain_core.prompts.structured import StructuredPrompt
from langsmith import traceable

from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Union,
)

import json


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

    messages[last_user_message_idx]["content"] += "\n\n" + "\n".join(  # type: ignore
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


def _construct_default_output_json_schema(
    *,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
) -> tuple[dict, str]:
    json_schema = {
        "type": "object",
        "additionalProperties": False,
    }
    # Set the description for the score schema
    if choices:
        description = "A number that represents the degree to which the criteria in the prompt are met."
        score_schema = {
            "type": "number",
            "description": description,
            "enum": choices,
        }
    elif continuous:
        description = "A number that represents the degree to which the criteria in the prompt are met, from 0.0 to 1.0. 1.0 means the criteria are met perfectly. 0.0 means none of the criteria are met, 0.5 means exactly half of the criteria are met."
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


def _stringify_prompt_param(param: Any) -> Any:
    if isinstance(param, str):
        return param
    elif isinstance(param, BaseMessage):
        return json.dumps(_convert_to_openai_message(param))
    elif isinstance(param, dict):
        if "messages" in param and isinstance(param["messages"], list):
            param["messages"] = [
                _convert_to_openai_message(msg) if isinstance(msg, BaseMessage) else msg
                for msg in param["messages"]
            ]
            return json.dumps(param)
        else:
            return json.dumps(param)
    elif isinstance(param, list):
        return json.dumps(
            [
                _convert_to_openai_message(msg) if isinstance(msg, BaseMessage) else msg
                for msg in param
            ]
        )
    else:
        return json.dumps(param)


def _create_llm_as_judge_scorer(
    *,
    prompt: str | Runnable | Callable[..., list[ChatCompletionMessage]],
    system: Optional[str] = None,
    schema: Optional[Union[dict, type]] = None,
    judge: Optional[
        Union[
            ModelClient,
            BaseChatModel,
        ]
    ] = None,
    model: Optional[str] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> Callable[..., ScoreType]:
    def get_score(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Optional[Union[str, dict]] = None,
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ):
        if system is not None and not isinstance(prompt, str):
            raise ValueError(
                "`system` is only supported when `prompt` is a string template"
            )

        stringified_outputs = (
            _stringify_prompt_param(outputs) if outputs is not None else None
        )
        stringified_reference_outputs = (
            _stringify_prompt_param(reference_outputs)
            if reference_outputs is not None
            else None
        )
        stringified_inputs = (
            _stringify_prompt_param(inputs) if inputs is not None else None
        )

        prompt_params = {
            "inputs": stringified_inputs,
            "outputs": stringified_outputs,
            "reference_outputs": stringified_reference_outputs,
            **kwargs,
        }
        filtered_prompt_params = {
            k: v for k, v in prompt_params.items() if v is not None
        }

        if isinstance(prompt, Runnable):
            formatted_prompt = prompt.invoke(filtered_prompt_params)
            messages = _normalize_to_openai_messages_list(formatted_prompt.messages)
            if isinstance(prompt, StructuredPrompt):
                nonlocal schema
                schema = prompt.schema_
        elif isinstance(prompt, str):
            formatted_prompt = prompt.format(**filtered_prompt_params)
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

        (default_json_schema, description) = _construct_default_output_json_schema(
            continuous=continuous,
            choices=choices,
            use_reasoning=use_reasoning,
        )

        nonlocal judge
        nonlocal model

        if judge is None:
            if model is None:
                raise ValueError("a `model` string is required (e.g. 'openai:o3-mini')")
            judge = init_chat_model(model=model)

        if isinstance(judge, BaseChatModel):
            judge_with_structured_output = judge.with_structured_output(
                schema
                if schema is not None
                else {
                    "title": "score",
                    "description": description,
                    **default_json_schema,
                }
            )
            response = judge_with_structured_output.invoke(messages)  # type: ignore
            if schema is None:
                if use_reasoning:
                    return (response["score"], response["reasoning"])  # type: ignore
                return response["score"]  # type: ignore
            else:
                return response
        elif isinstance(judge, ModelClient):
            if model is None:
                raise ValueError("a `model` string is required (e.g. 'openai:o3-mini')")
            if model.startswith("openai:"):
                model = model[len("openai:") :]

            openai_json_schema = default_json_schema
            if schema is not None:
                if not isinstance(schema, dict):
                    raise ValueError(
                        "`schema` must be JSON schema or OpenAI structured output format when using an OpenAI client directly"
                    )
                openai_json_schema = schema
            if openai_json_schema.get("name") is None:
                openai_json_schema = {
                    "name": "score",
                    "strict": True,
                    "schema": openai_json_schema,
                }
            if openai_json_schema.get("schema", {}).get("additionalProperties") is None:
                openai_json_schema["schema"]["additionalProperties"] = False

            params = {
                "messages": messages,
                "model": model,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": openai_json_schema,
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

            response = invoke_llm(**params)  # type: ignore
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
    prompt: str | Runnable | Callable[..., list[ChatCompletionMessage]],
    system: Optional[str] = None,
    schema: Optional[Union[dict, type]] = None,
    judge: Optional[
        Union[
            ModelClient,
            BaseChatModel,
        ]
    ] = None,
    model: Optional[str] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
) -> Callable[..., Union[Awaitable[ScoreType], ScoreType]]:
    async def aget_score(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Optional[Union[str, dict]] = None,
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ):
        if system is not None and not isinstance(prompt, str):
            raise ValueError(
                "`system` is only supported when `prompt` is a string template"
            )

        stringified_outputs = (
            _stringify_prompt_param(outputs) if outputs is not None else None
        )
        stringified_reference_outputs = (
            _stringify_prompt_param(reference_outputs)
            if reference_outputs is not None
            else None
        )
        stringified_inputs = (
            _stringify_prompt_param(inputs) if inputs is not None else None
        )

        prompt_params = {
            "inputs": stringified_inputs,
            "outputs": stringified_outputs,
            "reference_outputs": stringified_reference_outputs,
            **kwargs,
        }
        filtered_prompt_params = {
            k: v for k, v in prompt_params.items() if v is not None
        }

        if isinstance(prompt, Runnable):
            formatted_prompt = prompt.invoke(filtered_prompt_params)
            messages = _normalize_to_openai_messages_list(formatted_prompt.messages)
            if isinstance(prompt, StructuredPrompt):
                nonlocal schema
                schema = prompt.schema_
        elif isinstance(prompt, str):
            formatted_prompt = prompt.format(**filtered_prompt_params)
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

        (default_json_schema, description) = _construct_default_output_json_schema(
            continuous=continuous,
            choices=choices,
            use_reasoning=use_reasoning,
        )

        nonlocal judge
        nonlocal model

        if judge is None:
            if model is None:
                raise ValueError("a `model` string is required (e.g. 'openai:o3-mini')")
            judge = init_chat_model(model=model)

        if isinstance(judge, BaseChatModel):
            judge_with_structured_output = judge.with_structured_output(
                schema
                if schema is not None
                else {
                    "title": "score",
                    "description": description,
                    **default_json_schema,
                }
            )
            response = await judge_with_structured_output.ainvoke(messages)  # type: ignore
            if schema is None:
                if use_reasoning:
                    return (response["score"], response["reasoning"])  # type: ignore
                return response["score"]  # type: ignore
            else:
                return response
        elif isinstance(judge, ModelClient):
            if model is None:
                raise ValueError("a `model` string is required (e.g. 'openai:o3-mini')")
            if model.startswith("openai:"):
                model = model[len("openai:") :]

            openai_json_schema = default_json_schema
            if schema is not None:
                if not isinstance(schema, dict):
                    raise ValueError(
                        "`schema` must be JSON schema or OpenAI structured output format when using an OpenAI client directly"
                    )
                openai_json_schema = schema
            if openai_json_schema.get("name") is None:
                openai_json_schema = {
                    "name": "score",
                    "strict": True,
                    "schema": openai_json_schema,
                }
            if openai_json_schema.get("schema", {}).get("additionalProperties") is None:
                openai_json_schema["schema"]["additionalProperties"] = False

            params = {
                "messages": messages,
                "model": model,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": openai_json_schema,
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

            response = await ainvoke_llm(**params)  # type: ignore
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
    prompt: str | Runnable | Callable[..., list[ChatCompletionMessage]],
    feedback_key: str = "score",
    judge: Optional[Union[ModelClient, BaseChatModel]] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
    output_schema: Optional[Union[dict, type]] = None,
) -> Union[SimpleEvaluator, Callable[..., Any]]:
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
        model: Model identifier to use. If "judge" is an OpenAI client,
            this argument should be a model name directly. If "judge" is omitted, must be a valid
            LangChain model identifier. See `init_chat_model` docs for more details:
            https://python.langchain.com/docs/how_to/chat_models_universal_init/.
        system: Optional system message to prepend to the prompt.
        continuous: If True, score will be a float between 0 and 1. If False, score will be boolean. Defaults to False.
        choices: Optional list of specific float values the score must be chosen from.
        use_reasoning: If True, includes explanation for the score in the output. Defaults to True.
        few_shot_examples: Optional list of example evaluations to append to the prompt.
        output_schema: Optional JSON schema, Pydantic model, or TypedDict for the output of the evaluator.
            If you are using an OpenAI client directly, this field must be OpenAI structured output format or JSON schema if provided.
    Returns:
        A function that takes inputs, outputs, reference_outputs, and other kwargs, formats them into
        a prompt, invokes the judge, and returns an evaluation result.
        If `output_schema` is provided, the evaluator will be a callable function returning a dict conforming to the provided schema.

    Example:
        ```python
        from openevals.llm import create_llm_as_judge

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
    if output_schema is not None and isinstance(prompt, StructuredPrompt):
        raise ValueError(
            "You may not provide both an `output_schema` parameter and a LangChain prompt with output schema."
        )

    scorer = _create_llm_as_judge_scorer(
        prompt=prompt,
        judge=judge,
        system=system,
        model=model,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
        schema=output_schema,
    )

    def _wrapped_evaluator(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Optional[Union[str, dict]] = None,
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> EvaluatorResult | dict:
        run_name = (
            "llm_as_judge"
            if feedback_key == "score"
            else f"llm_as_{feedback_key}_judge"
        )
        res = _run_evaluator_untyped(
            run_name=run_name,
            scorer=scorer,
            feedback_key=feedback_key,
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            return_raw_outputs=output_schema is not None
            or isinstance(prompt, StructuredPrompt),
            **kwargs,
        )
        if output_schema is not None:
            return res  # type: ignore
        if isinstance(res, list):
            return res[0]
        return res  # type: ignore

    return _wrapped_evaluator  # type: ignore


def create_async_llm_as_judge(
    *,
    prompt: str | Runnable | Callable[..., list[ChatCompletionMessage]],
    feedback_key: str = "score",
    judge: Optional[Union[ModelClient, BaseChatModel]] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
    few_shot_examples: Optional[list[FewShotExample]] = None,
    output_schema: Optional[Union[dict, type]] = None,
) -> Union[SimpleAsyncEvaluator, Callable[..., Awaitable[Any]]]:
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
        model: Model identifier to use. If "judge" is an OpenAI client,
            this argument should be a model name directly. If "judge" is omitted, must be a valid
            LangChain model identifier. See `init_chat_model` docs for more details:
            https://python.langchain.com/docs/how_to/chat_models_universal_init/.
        system: Optional system message to prepend to the prompt.
        continuous: If True, score will be a float between 0 and 1. If False, score will be boolean. Defaults to False.
        choices: Optional list of specific float values the score must be chosen from.
        use_reasoning: If True, includes explanation for the score in the output. Defaults to True.
        few_shot_examples: Optional list of example evaluations to append to the prompt.
        output_schema: Optional JSON schema for the output of the evaluator.
            If you are using an OpenAI client directly, this field must be OpenAI structured output format or JSON schema if provided.
    Returns:
        A function that takes inputs, outputs, reference_outputs, and other kwargs, formats them into
        a prompt, invokes the judge, and returns an evaluation result.
        If `output_schema` is provided, the evaluator will be a callable function returning a dict conforming to the provided schema.

    Example:
        ```python
        from openevals.llm import create_async_llm_as_judge

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
    if output_schema is not None and isinstance(prompt, StructuredPrompt):
        raise ValueError(
            "You may not provide both an `output_schema` parameter and a LangChain prompt with output schema."
        )

    scorer = _create_async_llm_as_judge_scorer(
        prompt=prompt,
        judge=judge,
        system=system,
        model=model,
        continuous=continuous,
        choices=choices,
        use_reasoning=use_reasoning,
        few_shot_examples=few_shot_examples,
        schema=output_schema,
    )

    async def _wrapped_evaluator(
        *,
        inputs: Optional[Union[str, dict]] = None,
        outputs: Optional[Union[str, dict]] = None,
        reference_outputs: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> EvaluatorResult | dict:
        run_name = (
            "llm_as_judge"
            if feedback_key == "score"
            else f"llm_as_{feedback_key}_judge"
        )
        res = await _arun_evaluator_untyped(
            run_name=run_name,
            scorer=scorer,
            feedback_key=feedback_key,
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            return_raw_outputs=output_schema is not None
            or isinstance(prompt, StructuredPrompt),
            **kwargs,
        )
        if output_schema is not None:
            return res  # type: ignore
        if isinstance(res, list):
            return res[0]
        return res

    return _wrapped_evaluator  # type: ignore
