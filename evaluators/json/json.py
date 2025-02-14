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
    judge_rubric: Dict[str,str] = {},
    exclude_keys: list[str] = [],
    judge: Optional[
        Union[
            ModelClient,
            LangChainLikeModel,
        ]
    ] = None,
    model: str = "openai:gpt-4o",
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
        # how to deal with keys in reference output that don't show up in output

        scores = {}
        criteria = ""
        
        for key, value in outputs.items():
            if key in exclude_keys:
                continue
            if key not in reference_outputs:
                scores[key] = 0
                continue
            if key not in judge_rubric and reference_outputs[key] == value:
                scores[key] = 1
            elif key not in judge_rubric:
                scores[key] = 0
            else:
                criteria += f"Key: {key}, Criteria: {judge_rubric[key]}\n" 
                json_schema["properties"][key] = {
                    "type": "boolean",
                    "description": f"Does the output for key {key}, follow the criteria? {judge_rubric[key]}",
                }

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
            output_keys: Optional[str] = None,
            expected_output_keys: Optional[str] = None,
            criteria: Optional[str] = None,
        ):
            if scorer is not None:
                llm_scores = scorer(
                    outputs=output_keys,
                    reference_outputs=expected_output_keys,
                    criteria=criteria,
                )
                scores.update(llm_scores)
            score = None
            if aggregator == "average":
                score = sum(scores.values()) / len(scores)
            elif aggregator == "all":
                score = 0 if 0 in scores.values() else 1

            # If there is an aggregator, return a single result 
            if score is not None:
                return score
            else:
                results = {}
                for key, value in scores.items():
                    results[key] = value
                if len(results) == 1:
                    return list(results.values())[0]
                return results
        
        return _run_evaluator(
            run_name="json_match_judge",
            scorer=_scorer,
            feedback_key="structured_match_score",
            output_keys=output_keys,
            expected_output_keys=expected_output_keys,
            criteria=criteria,
            **kwargs,
        )

    return wrapped_evaluator