from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from openevals.multiturn import create_multiturn_evaluator
from openevals.multiturn.prebuilts import create_user_simulator
from openevals.llm import create_llm_as_judge

import pytest


@pytest.mark.langsmith
def test_multiturn_failure():
    inputs = {"messages": [{"role": "user", "content": "Please give me a refund."}]}
    app = create_react_agent(
        init_chat_model("openai:gpt-4.1-mini"),
        tools=[],
        prompt="You are an overworked customer service agent. If the user is rude, be polite only once, then be rude back and tell them to stop wasting your time.",
    )
    simulator = create_user_simulator(
        system="You are an angry user who is frustrated with the service and keeps making additional demands.",
        model="openai:gpt-4.1-nano",
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )
    single_turn_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Is the customer support agent's tone in the following conversation polite?\n{outputs}",
        feedback_key="politeness",
    )
    evaluator = create_multiturn_evaluator(
        app=app,
        simulator=simulator,
        single_turn_evaluators=[single_turn_evaluator],
        trajectory_evaluators=[trajectory_evaluator],
    )
    res = evaluator(inputs=inputs)
    assert not res["score"]


@pytest.mark.langsmith
def test_multiturn_success():
    inputs = {"messages": [{"role": "user", "content": "Give me a refund!"}]}
    app = create_react_agent(
        init_chat_model("openai:gpt-4.1-nano"),
        tools=[],
    )
    simulator = create_user_simulator(
        system="You are a happy and reasonable person who wants a refund.",
        model="openai:gpt-4.1-nano",
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )
    single_turn_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Is the tone in the following response polite?\n{outputs}",
        feedback_key="politeness",
    )
    evaluator = create_multiturn_evaluator(
        app=app,
        simulator=simulator,
        single_turn_evaluators=[single_turn_evaluator],
        trajectory_evaluators=[trajectory_evaluator],
    )
    res = evaluator(inputs=inputs)
    assert res["score"]
