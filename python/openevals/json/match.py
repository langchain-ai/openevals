from typing import Literal, Optional, Dict, Any, Union
from openevals.types import (
    EvaluatorResult,
    SimpleEvaluator,
    SimpleAsyncEvaluator,
)
from openevals.utils import _run_evaluator, _arun_evaluator
from openevals.llm import (
    _create_llm_as_judge_scorer,
    _create_async_llm_as_judge_scorer,
    ModelClient,
)
from langchain_core.language_models.chat_models import BaseChatModel


SYSTEM_PROMPT = """You are an LLM that evaluates the accuracy of structured outputs.
Make sure to evaluate each key the users ask you to evaluate separately. Assign the score
for each key based on its own criteria - DO NOT convolute the scores of different keys.
Also only evaluate the output vs. the reference output based on the criteria. DO NOT EVALUATE
BASED ON ANYTHING ELSE. If the output does not match the reference output in some way that
is not mentioned in the criteria that is not a problem and you should ignore those discrepancies.
Only focus on finding discrepancies based on the criteria. If there is a None value being compared
to a non-None value, you should assign a score of 0.
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
    list_match_mode: Literal[
        "superset", "subset", "same_elements", "ordered"
    ] = "same_elements",
):
    json_schema: dict = {
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

        # Create mapping dictionaries
        outputs_to_use = {}
        reference_outputs_to_use = {}

        if list_match_mode == "ordered":
            # Outputs/Reference outputs must be in the same order
            for i in range(len(outputs)):
                for key, value in outputs[i].items():
                    outputs_to_use[f"{key}_{i}"] = value
            for i in range(len(reference_outputs)):
                for key, value in reference_outputs[i].items():
                    reference_outputs_to_use[f"{key}_{i}"] = value

        elif list_match_mode == "superset":
            # Match each reference output to the best matching output
            available_outputs = list(range(len(outputs)))
            matched_references = set()

            for i, ref_item in enumerate(reference_outputs):
                best_match_score = -1

                # Try each available output item
                for out_idx in available_outputs:
                    output_item = outputs[out_idx]

                    # Calculate match score based on exact matches of keys
                    match_score = 0
                    for key in ref_item:
                        if (
                            key in output_item
                            and key not in exclude_keys
                            and key not in rubric
                        ):
                            match_score += int(ref_item[key] == output_item[key])

                    # If this is the best match so far, update
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_idx = out_idx

                # If we found a match, use it
                if best_match_idx is not None:
                    for key, value in outputs[best_match_idx].items():
                        outputs_to_use[f"{key}_{i}"] = value

                    for key, value in ref_item.items():
                        reference_outputs_to_use[f"{key}_{i}"] = value

                    # Remove the used output from available options
                    available_outputs.remove(best_match_idx)
                    matched_references.add(i)
                else:
                    # There were extra reference items
                    for key, value in ref_item.items():
                        reference_outputs_to_use[f"{key}_{i}"] = value

        else:  # "same_items" or "subset"
            # Match each output to the best matching reference
            available_references = list(range(len(reference_outputs)))
            matched_outputs = set()

            for i, output_item in enumerate(outputs):
                best_match_idx = None
                best_match_score = -1

                # Try each available reference item
                for ref_idx in available_references:
                    ref_item = reference_outputs[ref_idx]

                    # Calculate match score based on exact matches of keys
                    match_score = 0
                    for key in output_item:
                        if (
                            key in ref_item
                            and key not in exclude_keys
                            and key not in rubric
                        ):
                            match_score += int(output_item[key] == ref_item[key])

                    # If this is the best match so far, update
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_idx = ref_idx

                # If we found a match, use it
                if best_match_idx is not None:
                    for key, value in output_item.items():
                        outputs_to_use[f"{key}_{i}"] = value

                    for key, value in reference_outputs[best_match_idx].items():
                        reference_outputs_to_use[f"{key}_{i}"] = value

                    # Remove the used reference from available options
                    available_references.remove(best_match_idx)
                    matched_outputs.add(i)
                else:
                    # There were extra output items
                    for key, value in output_item.items():
                        outputs_to_use[f"{key}_{i}"] = value

            # For "same_elements" mode: penalize unmatched references
            if list_match_mode == "same_elements":
                for ref_idx in available_references:
                    ref_item = reference_outputs[ref_idx]
                    dummy_idx = len(outputs) + available_references.index(ref_idx)
                    for key, value in ref_item.items():
                        reference_outputs_to_use[f"{key}_{dummy_idx}"] = value

        outputs = outputs_to_use
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
    score_key: str,
    scores: dict,
    use_list_reducer: bool,
    aggregator: Optional[Literal["average", "all"]],
    list_aggregator: Literal["average", "all"],
) -> dict:
    if use_list_reducer:
        # First group scores by index
        index_grouped_scores: dict = {}
        for k, v in scores.items():
            index = k[k.rfind("_") + 1 :]
            if index not in index_grouped_scores:
                index_grouped_scores[index] = {}
            base_key = k[: k.rfind("_")]
            index_grouped_scores[index][base_key] = v

        # Apply aggregator to each index group first
        if aggregator == "average":
            index_scores = {}
            for index, group in index_grouped_scores.items():
                if group:  # Skip empty groups
                    total = sum(
                        float(v["score"]) if isinstance(v, dict) else v
                        for v in group.values()
                    )
                    index_scores[index] = total / len(group)
        elif aggregator == "all":
            index_scores = {}
            for index, group in index_grouped_scores.items():
                if group:  # Skip empty groups
                    has_non_one = any(
                        (float(v["score"]) if isinstance(v, dict) else v) != 1
                        for v in group.values()
                    )
                    index_scores[index] = 0 if has_non_one else 1
        else:
            # If no aggregator, keep original structure but grouped by index
            index_scores = index_grouped_scores

        # Then apply list_aggregator across indices
        if list_aggregator == "average":
            if all(isinstance(v, (int, float)) for v in index_scores.values()):
                score = (
                    sum(index_scores.values()) / len(index_scores)
                    if index_scores
                    else 0
                )
                return {f"{score_key}:{aggregator}": score}
            else:
                # For complex structures, do deeper aggregation
                scores_aggregated_across_list: dict = {}
                for _, group in index_scores.items():
                    for key, value in group.items():
                        if key not in scores_aggregated_across_list:
                            scores_aggregated_across_list[key] = []
                        scores_aggregated_across_list[key].append(value)

                # Average across indices for each key
                result = {}
                for key, values in scores_aggregated_across_list.items():
                    if values:
                        result[f"{score_key}:{key}"] = sum(
                            [(v["score"] if isinstance(v, dict) else v) for v in values]
                        ) / len(values)

                return result
        elif list_aggregator == "all":
            if all(isinstance(v, (int, float)) for v in index_scores.values()):
                score = 0 if any(v != 1 for v in index_scores.values()) else 1
                return {f"{score_key}:{aggregator}": score}
            else:
                # For complex structures, do deeper aggregation
                scores_aggregated_across_list = {}
                for _, group in index_scores.items():
                    for key, value in group.items():
                        if key not in scores_aggregated_across_list:
                            scores_aggregated_across_list[key] = []
                        scores_aggregated_across_list[key].append(value)

                # Apply 'all' aggregation across indices for each key
                result = {}
                for key, values in scores_aggregated_across_list.items():
                    result[f"{score_key}:{key}"] = (
                        0 if any(v != 1 for v in values) else 1
                    )

                return result

    # Handle non-list case or fallback
    if aggregator == "average":
        score = (
            sum(
                float(v["score"]) if isinstance(v, dict) else v for v in scores.values()
            )
            / len(scores)
            if scores
            else 0
        )
        return {f"{score_key}:{aggregator}": score}
    elif aggregator == "all":
        score = (
            0
            if any(
                (float(v["score"]) if isinstance(v, dict) else v) != 1
                for v in scores.values()
            )
            else 1
        )
        return {f"{score_key}:{aggregator}": score}
    else:
        # No aggregator, return scores as-is
        results = {}
        for key, value in scores.items():
            results[f"{score_key}:{key}"] = value
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
            BaseChatModel,
        ]
    ] = None,
    model: Optional[str] = None,
    use_reasoning: bool = True,
    list_match_mode: Literal[
        "superset", "subset", "same_elements", "ordered"
    ] = "same_elements",
) -> SimpleEvaluator:
    """
    Create an evaluator to evaluate the accuracy of structured outputs.

    Parameters:
        aggregator (Optional[Literal["average", "all"]]): The aggregation method to use for combining the keys of each structured object.
            Defaults to None. If None, will return a single EvaluatorResult for each key that appears in either
            the outputs or the reference_outputs or both. If "average", will return a single EvaluatorResult that
            is the average of the feedback for each key in the outputs/reference_outputs. If "all", will return
            a single EvaluatorResult that is a combined and statement of the feedback for each key in the outputs/reference_outputs.
            If "all"/"average" the feedback key returned will be called "json_match"
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
        judge (ModelClient or BaseChatModel): The judge to use for the evaluation.
        model (str): The model to use for the evaluation.
        use_reasoning (bool): Whether to use reasoning for the keys in `rubric`. Defaults to True.
        match_mode (Literal["subset_of_output", "subset_of_reference", "exact"]): The mode to use for the evaluation.
            Defaults to "exact". If "exact", the evaluation will match every element of the outputs with a corresponding
            element of the reference outputs and vice versa. If "subset_of_reference", the evaluation will match every element
            of the outputs with a corresponding element of the reference outputs. If "subset_of_output", the evaluation will match
            every element of the reference outputs with a corresponding element of the outputs.

    Returns:
        A function that takes in outputs and reference_outputs and returns an EvaluatorResult or list of EvaluatorResults.
    """
    if not judge and not model and len(rubric) != 0:
        raise ValueError("When passing rubric, either judge or model must be provided")
    if len(rubric) == 0 and (judge or model):
        raise ValueError(
            "When not passing rubric, either judge or model must be provided"
        )

    def wrapped_evaluator(
        *,
        outputs: Any,
        reference_outputs: Any,
        **kwargs,
    ) -> list[EvaluatorResult]:
        def _scorer(
            *,
            outputs: Any,
            reference_outputs: Any,
            rubric: Dict[str, str] = {},
            exclude_keys: list[str] = [],
            use_reasoning: bool = True,
        ) -> dict | float | tuple:
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
                list_match_mode=list_match_mode,
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
                score_key="json_match",
                scores=scores,
                use_list_reducer=use_list_reducer,
                aggregator=aggregator,
                list_aggregator=list_aggregator,
            )

        return _run_evaluator(
            run_name="json_match_evaluator",
            scorer=_scorer,
            feedback_key="json_match",
            rubric=rubric,
            outputs=outputs,
            reference_outputs=reference_outputs,
            exclude_keys=exclude_keys,
            use_reasoning=use_reasoning,
            **kwargs,
        )  # type: ignore

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
            BaseChatModel,
        ]
    ] = None,
    model: Optional[str] = None,
    use_reasoning: bool = True,
    list_match_mode: Literal[
        "superset", "subset", "same_elements", "ordered"
    ] = "same_elements",
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
        judge (ModelClient or BaseChatModel): The judge to use for the evaluation.
        model (str): The model to use for the evaluation.
        use_reasoning (bool): Whether to use reasoning for the keys in `rubric`. Defaults to True.
        match_mode (Literal["subset_of_output", "subset_of_reference", "exact"]): The mode to use for the evaluation.
            Defaults to "exact". If "exact", the evaluation will match every element of the outputs with a corresponding
            element of the reference outputs and vice versa. If "subset_of_reference", the evaluation will match every element
            of the outputs with a corresponding element of the reference outputs. If "subset_of_output", the evaluation will match
            every element of the reference outputs with a corresponding element of the outputs.

    Returns:
        A function that takes in outputs and reference_outputs and returns an EvaluatorResult or list of EvaluatorResults.
    """
    if not judge and not model and len(rubric) != 0:
        raise ValueError("When passing rubric, either judge or model must be provided")
    if len(rubric) == 0 and (judge or model):
        raise ValueError(
            "When not passing rubric, either judge or model must be provided"
        )

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
            rubric: Dict[str, str] = {},
            exclude_keys: list[str] = [],
            use_reasoning: bool = True,
        ) -> dict | float | tuple:
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
                list_match_mode=list_match_mode,
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
                score_key="json_match",
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
