import pytest
from evaluators.prebuilts import *

@pytest.mark.langsmith
def test_concise():
    inputs = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Who won the world series in 2020?",
        }
    ]
    outputs = {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020.",
    }
    evaluator = ConcisenessEvaluator()
    eval_result = evaluator(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None