import pytest

from openevals.llm import create_llm_as_judge
from openevals.prompts.correctness import CORRECTNESS_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_correctness():
    inputs = {
        "question": "Who was the first president of the United States?",
    }
    outputs = {"answer": "George Washington"}
    reference_outputs = {"answer": "George Washington"}
    llm_as_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        model="openai:o3-mini",
    )
    with pytest.raises(KeyError):
        eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    eval_result = llm_as_judge(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_correctness_not_correct():
    inputs = {
        "question": "Who was the first president of the United States?",
    }
    outputs = {"answer": "John Adams"}
    reference_outputs = {"answer": "George Washington"}
    llm_as_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    assert not eval_result["score"]
