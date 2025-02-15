import pytest

from evaluators.llm import create_llm_as_judge
from evaluators.prompts.correctness import CORRECTNESS_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_correctness():
    inputs = {
        "question": "Who was the first president of the United States?",
    }
    outputs = {"answer": "George Washington"}
    llm_as_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] == True


@pytest.mark.langsmith
def test_llm_as_judge_correctness_not_correct():
    inputs = {
        "question": "Who was the first president of the United States?",
    }
    outputs = {"answer": "John Adams"}
    llm_as_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] == False


@pytest.mark.langsmith
def test_llm_as_judge_correctness_with_reference_outputs():
    inputs = {
        "question": "Who was the first president of the Star Republic of Oiewjoie?",
    }
    outputs = {"answer": "Bzkeoei Ahbeijo"}
    llm_as_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
    )
    eval_result = llm_as_judge(
        inputs=inputs, outputs=outputs, reference_outputs={"answer": "Bzkeoei Ahbeijo"}
    )
    assert eval_result["score"] == True
