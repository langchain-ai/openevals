import pytest

from openevals.llm import create_llm_as_judge
from openevals.prompts.conciseness import CONCISENESS_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_conciseness():
    inputs = {
        "question": "How is the weather in San Francisco?",
    }
    outputs = {"answer": "Sunny and 90 degrees."}
    llm_as_judge = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        feedback_key="conciseness",
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_conciseness_not_concise():
    inputs = {
        "question": "How is the weather in San Francisco?",
    }
    outputs = {
        "answer": "Thanks for asking! The current weather in San Francisco is sunny and 90 degrees."
    }
    llm_as_judge = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        feedback_key="conciseness",
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]
