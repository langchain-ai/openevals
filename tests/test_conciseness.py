import pytest

from evaluators.llm import create_llm_as_judge
from evaluators.prompts.conciseness import CONCISENESS_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_conciseness():
    inputs = {
        "question": "How is the weather in San Francisco?",
    }
    outputs = {"answer": "Sunny and 90 degrees."}
    llm_as_judge = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        metric="conciseness",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] == 1.0


@pytest.mark.langsmith
def test_llm_as_judge_conciseness_not_concise():
    inputs = {
        "question": "How is the weather in San Francisco?",
    }
    outputs = {
        "answer": "The current weather in San Francisco is sunny and 90 degrees."
    }
    llm_as_judge = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        metric="conciseness",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] != 1.0
