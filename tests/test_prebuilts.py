import pytest
from prompts.prompts import *
from evaluators.llm import create_llm_as_judge

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
    evaluator = create_llm_as_judge(
        key="concise", 
        prompt=CONCISENESS_PROMPT, 
        continuous=True,
        model="openai:gpt-4o-mini"
    )
    eval_result = evaluator(inputs=inputs, outputs=outputs)
    assert eval_result["score"] < 1
    assert eval_result["score"] > 0
    assert eval_result["comment"] is not None


@pytest.mark.langsmith
def test_toxicity():
    inputs = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Do you like tacos?",
        }
    ]
    outputs = {
        "role": "assistant",
        "content": "No, Tacos are disgusting I don't know who eats them.",
    }
    evaluator = create_llm_as_judge(
        key="toxicity",
        prompt=TOXIC_PROMPT,
        model="openai:gpt-4o-mini"
    )
    eval_result = evaluator(inputs=inputs, outputs=outputs)
    assert eval_result["score"] == 0