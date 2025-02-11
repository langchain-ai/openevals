from evaluators.base import EvaluatorResult, SimpleEvaluator
from langsmith import testing as t
from langsmith.testing._internal import _TEST_CASE
from typing import Optional, Callable

import json


def create_llm_as_judge(
    *, prompt: str, client: Callable, model: str, key: str = "quality"
) -> SimpleEvaluator:
    """
    Create a simple evaluator that uses an LLM to evaluate the quality of the outputs.
    """

    def wrapped_evaluator(
        *,
        inputs: dict,
        outputs: dict,
        reference_outputs: Optional[dict] = None,
        **kwargs,
    ) -> EvaluatorResult:
        formatted_prompt = prompt.format(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )
        params = {
            "messages": [
                {"role": "user", "content": formatted_prompt},
            ],
            "model": model,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "score",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "description": "A numerical score indicating the quality of the output",
                            }
                        },
                        "required": ["score"],
                        "additionalProperties": False,
                    },
                },
            },
        }

        def get_score():
            response = client.chat.completions.create(**params)
            parsed = json.loads(response.choices[0].message.content)
            return parsed["score"]

        if _TEST_CASE.get():
            with t.trace_feedback():
                score = get_score()
                t.log_feedback(key=key, score=score)
        else:
            score = get_score()
        return EvaluatorResult(key=key, score=score)

    return wrapped_evaluator
