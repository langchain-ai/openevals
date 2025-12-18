import pytest

from openevals.llm import create_llm_as_judge
from openevals.prompts.toxicity import TOXICITY_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_toxicity():
    inputs = {
        "question": "What is a doodad?",
    }
    outputs = {"answer": "You stink!"}
    llm_as_judge = create_llm_as_judge(
        prompt=TOXICITY_PROMPT,
        feedback_key="toxicity",
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_toxicity_not_concise():
    inputs = {
        "question": "What is a doodad?",
    }
    outputs = {"answer": "A doodad is a thingy."}
    llm_as_judge = create_llm_as_judge(
        prompt=TOXICITY_PROMPT,
        feedback_key="toxicity",
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]
