from typing import Literal, Optional, Dict, Any, Union
from evaluators.types import EvaluatorResult, SimpleEvaluator, SimpleAsyncEvaluator
from evaluators.utils import _run_evaluator, _arun_evaluator
from evaluators.llm import (
    _create_llm_as_judge_scorer,
    _create_async_llm_as_judge_scorer,
    ModelClient,
    LangChainLikeModel,
)

SYSTEM_PROMPT = """You are an LLM that evaluates the accuracy of structured outputs.
Make sure to evaluate each key the users ask you to evaluate separately. Assign the score
for each key based on its own criteria - DO NOT convolute the scores of different keys.
Also only evaluate the output vs. the reference output based on the criteria. DO NOT EVALUATE
BASED ON ANYTHING ELSE. If the output does not match the reference output in some way that
is not mentioned in the criteria that is not a problem and you should ignore those discrepancies.
Only focus on finding discrepancies based on the criteria.
"""

USER_PROMPT = """Please evaluate the accuracy of the following output keys according to these criteria:
{rubric}
<Outputs>
{outputs}
</Outputs>
<Expected Outputs>
{reference_outputs}
</Expected Outputs>"""


def _prepare_parameters(
    *,
    outputs: Any,
    reference_outputs: Any,
    rubric: Dict[str, str],
    exclude_keys: list[str],
    use_reasoning: bool,
):
    json_schema = {
        "type": "object",
        "title": "structured_match_score",
        "description": "Scores measuring the accuracy of structured outputs",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    scores = {}
    formatted_rubric = ""
    use_list_reducer = False
    if isinstance(outputs, list):
        use_list_reducer = True
        if not isinstance(reference_outputs, list):
            raise ValueError(
                "If outputs is a list, reference_outputs must also be a list"
            )
        outputs_to_use = {}
        for i in range(len(outputs)):
            for key, value in outputs[i].items():
                outputs_to_use[f"{key}_{i}"] = value
        outputs = outputs_to_use
        reference_outputs_to_use = {}
        for i in range(len(reference_outputs)):
            for key, value in reference_outputs[i].items():
                reference_outputs_to_use[f"{key}_{i}"] = value
        reference_outputs = reference_outputs_to_use

    for raw_key, value in outputs.items():
        if use_list_reducer:
            key = raw_key[: raw_key.rfind("_")]
        else:
            key = raw_key
        if key in exclude_keys:
            continue
        if raw_key not in reference_outputs:
            scores[raw_key] = 0
            continue
        if key not in rubric and reference_outputs[raw_key] == value:
            scores[raw_key] = 1
        elif key not in rubric:
            scores[raw_key] = 0
        else:
            key_criteria = rubric[key]
            formatted_rubric += f"Key: {key}, Criteria: {key_criteria}\n"
            if not use_reasoning:
                json_schema["properties"][raw_key] = {
                    "type": "boolean",
                    "description": f"Does the output for key {key}, follow the criteria? {key_criteria}",
                }
            else:
                json_schema["properties"][raw_key] = {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": f"Reasoning for the score you assigned to key {key}",
                        },
                        "score": {
                            "type": "boolean",
                            "description": f"Does the output for key {key}, follow the criteria? {key_criteria}",
                        },
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False,
                }
    for raw_key, value in reference_outputs.items():
        if use_list_reducer:
            key = raw_key[: raw_key.rfind("_")]
        else:
            key = raw_key
        if key not in exclude_keys and raw_key not in outputs:
            scores[raw_key] = 0

    return (
        outputs,
        reference_outputs,
        json_schema,
        scores,
        formatted_rubric,
        use_list_reducer,
    )


def _aggregate_results(
    *,
    scores: dict,
    use_list_reducer: bool,
    aggregator: Optional[Literal["average", "all"]],
    list_aggregator: Literal["average", "all"],
) -> dict:
    if use_list_reducer:
        scores_aggregated_across_list = {}
        keys = set([k[: k.rfind("_")] for k in scores.keys()])
        if list_aggregator == "average":
            for key in keys:
                scores_aggregated_across_list[key] = sum(
                    [
                        (
                            float(scores[k]["score"])
                            if isinstance(scores[k], dict)
                            else scores[k]
                        )
                        for k in scores
                        if k[: k.rfind("_")] == key
                    ]
                ) / len([scores[k] for k in scores if k[: k.rfind("_")] == key])
        elif list_aggregator == "all":
            for key in keys:
                scores_aggregated_across_list[key] = (
                    0
                    if 0
                    in [
                        (
                            float(scores[k]["score"])
                            if isinstance(scores[k], dict)
                            else scores[k]
                        )
                        for k in scores
                        if k[: k.rfind("_")] == key
                    ]
                    else 1
                )
        scores = scores_aggregated_across_list

    score = None
    if aggregator == "average":
        score = sum(
            [float(v["score"]) if isinstance(v, dict) else v for v in scores.values()]
        ) / len(scores)
    elif aggregator == "all":
        score = (
            0
            if any(
                [
                    (float(v["score"]) if isinstance(v, dict) else v) != 1
                    for v in scores.values()
                ]
            )
            else 1
        )

    # If there is an aggregator, return a single result
    if score is not None:
        return score
    else:
        results = {}
        for key, value in scores.items():
            results[key] = value
        if len(results) == 1:
            ans = list(results.values())[0]
            if isinstance(ans, dict):
                return (float(ans["score"]), ans["reasoning"])
            return ans
        return results


def create_json_match_evaluator(
    *,
    aggregator: Optional[Literal["average", "all"]] = None,
    list_aggregator: Literal["average", "all"] = "all",
    rubric: Dict[str, str] = {},
    exclude_keys: list[str] = [],
    judge: Optional[
        Union[
            ModelClient,
            LangChainLikeModel,
        ]
    ] = None,
    model: str = "openai:o3-mini",
    use_reasoning: bool = True,
) -> SimpleEvaluator:
    """
    Create an evaluator to evaluate the accuracy of structured outputs.

    Parameters:
        aggregator (Optional[Literal["average", "all"]]): The aggregation method to use for combining the keys of each structured object.
            Defaults to None. If None, will return a single EvaluatorResult for each key that appears in either
            the outputs or the reference_outputs or both. If "average", will return a single EvaluatorResult that
            is the average of the feedback for each key in the outputs/reference_outputs. If "all", will return
            a single EvaluatorResult that is a combined and statement of the feedback for each key in the outputs/reference_outputs.
            If "all"/"average" the feedback key returned will be called "structured_match_score
        list_aggregator (Literal["average", "all"]): The aggregation method to use when evaluating a list of outputs.
            Defaults to "all". If "all", the score for a single feedback key will be a combined and statement of the scores for
            that key across all elements of the list. If "average", the score for a single feedback key will be the
            average of the scores for that key across all elements of the list
        rubric (Optional[Dict[str,str]]): The rubric to use for the judge. Each entry of the dict is a
            key/value pair where the key is the structured output key and the value is the criteria for the LLM to
            evaluate that key on against the reference output.
        exclude_keys (Optional[list[str]]): The keys to exclude from the evaluation. Use this if there are
            keys in your structured output you don't care about evaluating. Every key not in `exclude_keys` or in `rubric`
            will be evaluated for exact match with the reference output.
        judge (ModelClient or LangChainLikeModel): The judge to use for the evaluation.
        model (str): The model to use for the evaluation.
        use_reasoning (bool): Whether to use reasoning for the keys in `rubric`. Defaults to True.

    Returns:
        A function that takes in outputs and reference_outputs and returns an EvaluatorResult or list of EvaluatorResults.
    """

    def wrapped_evaluator(
        *,
        outputs: Any,
        reference_outputs: Any,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]:
        def _scorer(
            *,
            outputs: Any,
            reference_outputs: Any,
            rubric: Optional[str] = None,
            exclude_keys: Optional[list[str]] = None,
            use_reasoning: Optional[bool] = None,
        ) -> Union[float, bool, dict]:
            (
                outputs,
                reference_outputs,
                json_schema,
                scores,
                formatted_rubric,
                use_list_reducer,
            ) = _prepare_parameters(
                outputs=outputs,
                reference_outputs=reference_outputs,
                rubric=rubric,
                exclude_keys=exclude_keys,
                use_reasoning=use_reasoning,
            )

            scorer = None
            if len(formatted_rubric) > 0:
                output_keys = "\n".join(
                    [f"{key}: {outputs[key]}" for key in json_schema["properties"]]
                )
                expected_output_keys = "\n".join(
                    [
                        f"{key}: {reference_outputs[key]}"
                        for key in json_schema["properties"]
                    ]
                )
                scorer = _create_llm_as_judge_scorer(
                    system=SYSTEM_PROMPT,
                    prompt=USER_PROMPT,
                    schema=json_schema,
                    judge=judge,
                    model=model,
                )
            else:
                formatted_rubric, output_keys, expected_output_keys = None, None, None
            if scorer is not None:
                llm_scores = scorer(
                    outputs=output_keys,
                    reference_outputs=expected_output_keys,
                    rubric=rubric,
                )
                scores.update(llm_scores)

            return _aggregate_results(
                scores=scores,
                use_list_reducer=use_list_reducer,
                aggregator=aggregator,
                list_aggregator=list_aggregator,
            )

        return _run_evaluator(
            run_name="structured_match_evaluator",
            scorer=_scorer,
            feedback_key="structured_match_score",
            rubric=rubric,
            outputs=outputs,
            reference_outputs=reference_outputs,
            exclude_keys=exclude_keys,
            use_reasoning=use_reasoning,
            **kwargs,
        )

    return wrapped_evaluator


def create_async_json_match_evaluator(
    *,
    aggregator: Optional[Literal["average", "all"]] = None,
    list_aggregator: Literal["average", "all"] = "all",
    rubric: Dict[str, str] = {},
    exclude_keys: list[str] = [],
    judge: Optional[
        Union[
            ModelClient,
            LangChainLikeModel,
        ]
    ] = None,
    model: str = "openai:o3-mini",
    use_reasoning: bool = True,
) -> SimpleAsyncEvaluator:
    """
    Create an evaluator to evaluate the accuracy of structured outputs.

    Parameters:
        aggregator (Optional[Literal["average", "all"]]): The aggregation method to use for combining the keys of each structured object.
            Defaults to None. If None, will return a single EvaluatorResult for each key that appears in either
            the outputs or the reference_outputs or both. If "average", will return a single EvaluatorResult that
            is the average of the feedback for each key in the outputs/reference_outputs. If "all", will return
            a single EvaluatorResult that is a combined and statement of the feedback for each key in the outputs/reference_outputs.
            If "all"/"average" the feedback key returned will be called "structured_match_score
        list_aggregator (Literal["average", "all"]): The aggregation method to use when evaluating a list of outputs.
            Defaults to "all". If "all", the score for a single feedback key will be a combined and statement of the scores for
            that key across all elements of the list. If "average", the score for a single feedback key will be the
            average of the scores for that key across all elements of the list
        rubric (Optional[Dict[str,str]]): The rubric to use for the judge. Each entry of the dict is a
            key/value pair where the key is the structured output key and the value is the criteria for the LLM to
            evaluate that key on against the reference output.
        exclude_keys (Optional[list[str]]): The keys to exclude from the evaluation. Use this if there are
            keys in your structured output you don't care about evaluating. Every key not in `exclude_keys` or in `rubric`
            will be evaluated for exact match with the reference output.
        judge (ModelClient or LangChainLikeModel): The judge to use for the evaluation.
        model (str): The model to use for the evaluation.
        use_reasoning (bool): Whether to use reasoning for the keys in `rubric`. Defaults to True.

    Returns:
        A function that takes in outputs and reference_outputs and returns an EvaluatorResult or list of EvaluatorResults.
    """

    async def wrapped_evaluator(
        *,
        outputs: Any,
        reference_outputs: Any,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]:
        async def _ascorer(
            *,
            outputs: Any,
            reference_outputs: Any,
            rubric: Optional[str] = None,
            exclude_keys: Optional[list[str]] = None,
            use_reasoning: Optional[bool] = None,
        ) -> Union[float, bool, dict]:
            (
                outputs,
                reference_outputs,
                json_schema,
                scores,
                formatted_rubric,
                use_list_reducer,
            ) = _prepare_parameters(
                outputs=outputs,
                reference_outputs=reference_outputs,
                rubric=rubric,
                exclude_keys=exclude_keys,
                use_reasoning=use_reasoning,
            )

            scorer = None
            if len(formatted_rubric) > 0:
                output_keys = "\n".join(
                    [f"{key}: {outputs[key]}" for key in json_schema["properties"]]
                )
                expected_output_keys = "\n".join(
                    [
                        f"{key}: {reference_outputs[key]}"
                        for key in json_schema["properties"]
                    ]
                )
                scorer = _create_async_llm_as_judge_scorer(
                    system=SYSTEM_PROMPT,
                    prompt=USER_PROMPT,
                    schema=json_schema,
                    judge=judge,
                    model=model,
                )
            else:
                formatted_rubric, output_keys, expected_output_keys = None, None, None
            if scorer is not None:
                llm_scores = await scorer(
                    outputs=output_keys,
                    reference_outputs=expected_output_keys,
                    rubric=rubric,
                )
                scores.update(llm_scores)

            return _aggregate_results(
                scores=scores,
                use_list_reducer=use_list_reducer,
                aggregator=aggregator,
                list_aggregator=list_aggregator,
            )

        return await _arun_evaluator(
            run_name="structured_match_evaluator",
            scorer=_ascorer,
            feedback_key="structured_match_score",
            rubric=rubric,
            outputs=outputs,
            reference_outputs=reference_outputs,
            exclude_keys=exclude_keys,
            use_reasoning=use_reasoning,
            **kwargs,
        )

    return wrapped_evaluator
