from typing import Literal, Optional, Dict, Any
from evaluators.types import EvaluatorResult, SimpleEvaluator
from langchain.chat_models import init_chat_model
from langsmith import testing as t
from langsmith.testing._internal import _TEST_CASE

META_PROMPT = """You an LLM that evaluates the accuracy of structured outputs.
Make sure to evaluate each key the users you to evaluate separately. Assign the score
for each key based on it's own criteria - DO NOT convolute the scores of different keys.
"""

def json_match_evaluator(
    *,
    aggregator: Optional[Literal["average", "all"]] = None,
    judge_rubric: Dict[str,str] = {},
    exclude_keys: list[str] = [],
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
        messages = [
            {"role": "system", "content": META_PROMPT},
        ]
        llm_judge_prompt_prefix = "Can you please evaluate the accuracy of the following output keys?\n"
        llm_judge_prompt = ""
        judge = init_chat_model(model=model)
        
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
                llm_judge_prompt += f"Key: {key}, Criteria: {judge_rubric[key]}\n" 
                json_schema["properties"][key] = {
                    "type": "boolean",
                    "description": f"Does the output for key {key}, follow the criteria? {judge_rubric[key]}",
                }

        if len(llm_judge_prompt) > 0:
            llm_judge_prompt = llm_judge_prompt_prefix + llm_judge_prompt
            output_keys = "\n".join([f"{key}: {outputs[key]}" for key in json_schema["properties"]])
            expected_output_keys = "\n".join([f"{key}: {reference_outputs[key]}" for key in json_schema["properties"]])
            llm_judge_prompt += f"""<Outputs>
            {output_keys}
            </Outputs>
            <Expected Outputs>
            {expected_output_keys}
            </Expected Outputs>"""
            messages.append({"role": "user", "content": llm_judge_prompt})
            judge = judge.with_structured_output(json_schema)
            key_scores = judge.invoke(messages)
            for key, score in key_scores.items():
                scores[key] = int(score)

        score = None
        if aggregator == "average":
            score = sum(scores.values()) / len(scores)
        elif aggregator == "all":
            score = 0 if 0 in scores.values() else 1

        # Otherwise we are returning per key scores 
        if score is not None:
            results = EvaluatorResult(key="structured_match_score", score=score)
        else:
            results = []
            for key, value in scores.items():
                results.append(EvaluatorResult(key=key, score=value))
            if len(results) == 1:
                results = results[0]

        if _TEST_CASE.get():
            with t.trace_feedback(name="json_match_evaluator"):
                if isinstance(results, list):
                    for result in results:
                        t.log_feedback(key=result["key"], score=result["score"])
                else:
                    t.log_feedback(key=results["key"], score=results["score"])
        return results

    return wrapped_evaluator