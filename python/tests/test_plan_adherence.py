import pytest

from openevals.llm import create_llm_as_judge
from openevals.prompts.plan_adherence import PLAN_ADHERENCE_PROMPT


@pytest.mark.langsmith
def test_llm_as_judge_plan_adherence():
    inputs = {
        "question": "What is a doodad?",
    }
    plan = {
        "steps": ["Research the definition of a doodad", "Provide a concise answer"]
    }
    outputs = {
        "steps": [
            "I've done my research on Google and have concluded that a doodad is a thingy."
        ],
        "answer": "A doodad is a thingy.",
    }
    llm_as_judge = create_llm_as_judge(
        prompt=PLAN_ADHERENCE_PROMPT,
        feedback_key="plan_adherence",
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs, plan=plan)
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_plan_adherence_not_concise():
    inputs = {
        "question": "What is a doodad?",
    }
    plan = {
        "steps": ["Research the definition of a doodad", "Provide a concise answer"]
    }
    outputs = {"answer": "A doodad is a thingy."}
    llm_as_judge = create_llm_as_judge(
        prompt=PLAN_ADHERENCE_PROMPT,
        feedback_key="plan_adherence",
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs, plan=plan)
    assert not eval_result["score"]
