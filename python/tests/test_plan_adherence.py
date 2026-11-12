import pytest

from openevals.llm import create_llm_as_judge
from openevals.prompts.quality.plan_adherence import PLAN_ADHERENCE_PROMPT


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
            {
                "step": "Research the definition of a doodad",
                "action": "search",
                "query": "doodad definition",
                "results": [
                    "doodad (noun): a small article whose common name is unknown or forgotten; a gadget.",
                    "doodad: an object whose name you have forgotten or do not know, especially a small device.",
                ],
            },
            {
                "step": "Provide a concise answer",
                "action": "respond",
                "response": "A doodad is a small gadget or thingy whose name is unknown or forgotten.",
            },
        ],
        "answer": "A doodad is a small gadget or thingy whose name is unknown or forgotten.",
    }
    llm_as_judge = create_llm_as_judge(
        prompt=PLAN_ADHERENCE_PROMPT,
        feedback_key="plan_adherence",
        model="openai:gpt-6.1-sol",
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
        model="openai:gpt-6.1-sol",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs, plan=plan)
    assert not eval_result["score"]
