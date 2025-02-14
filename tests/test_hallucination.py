import pytest

from evaluators.llm import create_llm_as_judge
from evaluators.prompts.hallucination import HALLUCINATION_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_correctness():
    inputs = {
        "question": "Who was the first president of the Star Republic of Oiewjoie?",
    }
    outputs = {"answer": "Bzkeoei Ahbeijo"}
    llm_as_judge = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        metric="hallucination",
    )
    context = "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo."
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs, context=context)
    assert eval_result["score"] == 1.0


@pytest.mark.langsmith
def test_llm_as_judge_correctness_not_correct():
    inputs = {
        "question": "Who was the first president of the Star Republic of Oiewjoie?",
    }
    outputs = {"answer": "John Adams"}
    llm_as_judge = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        metric="hallucination",
    )
    context = "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo."
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs, context=context)
    assert eval_result["score"] != 1.0
