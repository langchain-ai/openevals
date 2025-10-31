import pytest

from openevals.llm import create_llm_as_judge
from openevals.prompts.hallucination import HALLUCINATION_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_hallucination():
    inputs = {
        "question": "Who was the first president of the Star Republic of Oiewjoie?",
    }
    outputs = {"answer": "Bzkeoei Ahbeijo"}
    llm_as_judge = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        feedback_key="hallucination",
        model="openai:o3-mini",
    )
    context = "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo."
    with pytest.raises(KeyError):
        eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    eval_result = llm_as_judge(
        inputs=inputs, outputs=outputs, context=context, reference_outputs=""
    )
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_hallucination_not_correct():
    inputs = {
        "question": "Who was the first president of the Star Republic of Oiewjoie?",
    }
    outputs = {"answer": "John Adams"}
    llm_as_judge = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        feedback_key="hallucination",
        model="openai:o3-mini",
    )
    context = "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo."
    eval_result = llm_as_judge(
        inputs=inputs, outputs=outputs, context=context, reference_outputs=""
    )
    assert not eval_result["score"]
