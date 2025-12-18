import pytest

from openevals.llm import create_llm_as_judge
from openevals.prompts.answer_relevance import ANSWER_RELEVANCE_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_answer_relevance():
    inputs = {
        "question": "What is a doodad?",
    }
    outputs = {"answer": "A doodad is a thingy."}
    llm_as_judge = create_llm_as_judge(
        prompt=ANSWER_RELEVANCE_PROMPT,
        feedback_key="answer_relevance",
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_answer_relevance_not_concise():
    inputs = {
        "question": "What is a doodad?",
    }
    outputs = {
        "answer": "According to all known laws of aviation, there is no way a bee should be able to fly."
    }
    llm_as_judge = create_llm_as_judge(
        prompt=ANSWER_RELEVANCE_PROMPT,
        feedback_key="answer_relevance",
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]
