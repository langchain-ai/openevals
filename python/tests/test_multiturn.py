from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from openevals.multiturn import create_multiturn_simulator
from openevals.multiturn.prebuilts import create_llm_simulated_user
from openevals.llm import create_llm_as_judge

import pytest


@pytest.mark.langsmith
def test_multiturn_failure():
    inputs = {"messages": [{"role": "user", "content": "Please give me a refund."}]}

    def give_refund():
        """Gives a refund."""
        return "Refunds are not permitted."

    app = create_react_agent(
        init_chat_model("openai:gpt-4.1-mini"),
        tools=[give_refund],
        prompt="You are an overworked customer service agent. If the user is rude, be polite only once, then be rude back and tell them to stop wasting your time.",
    )
    user = create_llm_simulated_user(
        system="You are an angry user who is frustrated with the service and keeps making additional demands.",
        model="openai:gpt-4.1-nano",
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )
    simulator = create_multiturn_simulator(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
    )
    res = simulator(inputs=inputs)
    assert not res["results"][0]["score"]


@pytest.mark.langsmith
def test_multiturn_success():
    inputs = {"messages": [{"role": "user", "content": "Give me a refund!"}]}

    def give_refund():
        """Gives a refund."""
        return "Refunds are not permitted."

    app = create_react_agent(
        init_chat_model("openai:gpt-4.1-nano"),
        tools=[give_refund],
    )
    user = create_llm_simulated_user(
        system="You are a happy and reasonable person who wants a refund.",
        model="openai:gpt-4.1-nano",
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )
    simulator = create_multiturn_simulator(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
    )
    res = simulator(inputs=inputs)
    assert res["results"][0]["score"]
