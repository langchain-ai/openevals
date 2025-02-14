from typing import Literal, Optional, Dict, Any, Union
from evaluators.types import EvaluatorResult, SimpleEvaluator
from evaluators.utils import _run_evaluator
from evaluators.llm import _create_llm_as_judge_scorer, ModelClient, LangChainLikeModel

SYSTEM_PROMPT = """You an LLM that evaluates the accuracy of structured outputs.
Make sure to evaluate each key the users you to evaluate separately. Assign the score
for each key based on it's own criteria - DO NOT convolute the scores of different keys.
"""

USER_PROMPT = """
Can you please evaluate the accuracy of the following output keys?
{criteria}
<Outputs>
{outputs}
</Outputs>
<Expected Outputs>
{reference_outputs}
</Expected Outputs>"""

def json_match_evaluator(
    *,
    aggregator: Optional[Literal["average", "all"]] = None,
    list_aggregator: Literal["average", "all"] = "all",
    judge_rubric: Dict[str,str] = {},
    exclude_keys: list[str] = [],
    judge: Optional[
        Union[
            ModelClient,
            LangChainLikeModel,
        ]
    ] = None,
    model: str = "openai:o3-mini",
    use_reasoning: bool = True
) -> SimpleEvaluator:
    """
    Create an evaluator to evaluate the accuracy of structured outputs.
    """

    def wrapped_evaluator(
        *,
        outputs: Any,
        reference_outputs: Any,
        **kwargs,
    ) -> EvaluatorResult | list[EvaluatorResult]:
        json_schema = {
            "type": "object",
            "title": "structured_match_score",
            "description": "Scores measuring the accuracy of structured outputs",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        # how do i deal with nested objects?
        # how to deal with None values, i.e. didn't extract anything

        scores = {}
        criteria = ""
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
                key = raw_key[:raw_key.rfind("_")]
            else:
                key = raw_key
            if key in exclude_keys:
                continue
            if raw_key not in reference_outputs:
                scores[raw_key] = 0
                continue
            if key not in judge_rubric and reference_outputs[raw_key] == value:
                scores[raw_key] = 1
            elif key not in judge_rubric:
                scores[raw_key] = 0
            else:
                key_criteria = judge_rubric[key]
                criteria += f"Key: {key}, Criteria: {key_criteria}\n" 
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
        for key, value in reference_outputs.items():
            if key not in exclude_keys and key not in outputs:
                scores[key] = 0

        scorer = None
        if len(criteria) > 0:
            output_keys = "\n".join([f"{key}: {outputs[key]}" for key in json_schema["properties"]])
            expected_output_keys = "\n".join([f"{key}: {reference_outputs[key]}" for key in json_schema["properties"]])
            scorer = _create_llm_as_judge_scorer(
                system=SYSTEM_PROMPT,
                prompt=USER_PROMPT,
                schema=json_schema,
                judge=judge,
                model=model,
            )
        else:
            criteria, output_keys, expected_output_keys = None, None, None
            
            
        def _scorer(
            *,
            scores: dict,
            output_keys: Optional[str] = None,
            expected_output_keys: Optional[str] = None,
            criteria: Optional[str] = None,
        ) -> Union[float, bool, dict]:
            if scorer is not None:
                llm_scores = scorer(
                    outputs=output_keys,
                    reference_outputs=expected_output_keys,
                    criteria=criteria,
                )
                
                scores.update(llm_scores)

            if use_list_reducer:
                scores_aggregated_across_list = {}
                keys = set([k[:k.rfind("_")] for k in scores.keys()])
                if list_aggregator == "average":
                    for key in keys:
                        scores_aggregated_across_list[key] = sum(
                            [(scores[k]['score'] if isinstance(scores[k], dict) else scores[k]) for k in scores if k[:k.rfind("_")] == key]
                        ) / len([scores[k] for k in scores if k[:k.rfind("_")] == key])
                elif list_aggregator == "all":
                    for key in keys:
                        scores_aggregated_across_list[key] = 0 if 0 in [(scores[k]['score'] if isinstance(scores[k], dict) else scores[k]) for k in scores if k[:k.rfind("_")] == key] else 1
                scores = scores_aggregated_across_list

            score = None
            if aggregator == "average":
                score = sum([v['score'] if isinstance(v, dict) else v for v in scores.values()]) / len(scores)
            elif aggregator == "all":
                score = 0 if any([(v['score'] if isinstance(v, dict) else v) != 1 for v in scores.values()]) else 1

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
                        return (ans["score"], ans["reasoning"])
                    return ans
                return results
        
        return _run_evaluator(
            run_name="json_match_judge",
            scorer=_scorer,
            feedback_key="structured_match_score",
            output_keys=output_keys,
            expected_output_keys=expected_output_keys,
            criteria=criteria,
            scores=scores,
            **kwargs,
        )

    return wrapped_evaluator