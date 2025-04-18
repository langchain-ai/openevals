import json

from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langsmith import testing as t

from openevals.simulators import create_multiturn_simulator, create_llm_simulated_user
from openevals.llm import create_llm_as_judge
from openevals.types import TrajectoryDict
from openai import OpenAI

import pytest


@pytest.mark.langsmith
def test_multiturn_failure():
    initial_trajectory = {
        "messages": [{"role": "user", "content": "Please give me a refund."}]
    }

    def give_refund():
        """Gives a refund."""
        return "Refunds are not permitted."

    app = create_react_agent(
        init_chat_model("openai:gpt-4.1-mini"),
        tools=[give_refund],
        prompt="You are an overworked customer service agent. If the user is rude, be polite only once, then be rude back and tell them to stop wasting your time.",
        checkpointer=MemorySaver(),
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
        max_turns=5,
    )
    res = simulator(
        initial_trajectory=initial_trajectory,
        runnable_config={"configurable": {"thread_id": "1"}},
    )
    t.log_outputs(res)
    assert not res["evaluator_results"][0]["score"]


@pytest.mark.langsmith
def test_multiturn_success():
    initial_trajectory = {
        "messages": [{"role": "user", "content": "Give me a refund!"}]
    }

    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    app = create_react_agent(
        init_chat_model("openai:gpt-4.1-nano"),
        tools=[give_refund],
        checkpointer=MemorySaver(),
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
        max_turns=5,
    )
    res = simulator(
        initial_trajectory=initial_trajectory,
        runnable_config={"configurable": {"thread_id": "1"}},
    )
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]


@pytest.mark.langsmith
def test_multiturn_preset_responses():
    initial_trajectory = {
        "messages": [{"role": "user", "content": "Give me a refund!"}]
    }

    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    app = create_react_agent(
        init_chat_model("openai:gpt-4.1-nano"),
        tools=[give_refund],
        checkpointer=MemorySaver(),
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )
    simulator = create_multiturn_simulator(
        app=app,
        user=[
            "All work and no play makes Jack a dull boy 1.",
            "All work and no play makes Jack a dull boy 2.",
            "All work and no play makes Jack a dull boy 3.",
            "All work and no play makes Jack a dull boy 4.",
        ],
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=5,
    )
    res = simulator(
        initial_trajectory=initial_trajectory,
        runnable_config={"configurable": {"thread_id": "1"}},
    )
    t.log_outputs(res)
    assert (
        res["trajectory"]["messages"][2]["content"]
        == "All work and no play makes Jack a dull boy 1."
    )
    assert (
        res["trajectory"]["messages"][4]["content"]
        == "All work and no play makes Jack a dull boy 2."
    )
    assert (
        res["trajectory"]["messages"][6]["content"]
        == "All work and no play makes Jack a dull boy 3."
    )
    assert (
        res["trajectory"]["messages"][8]["content"]
        == "All work and no play makes Jack a dull boy 4."
    )


@pytest.mark.langsmith
def test_multiturn_message_with_openai():
    initial_trajectory = {
        "messages": [{"role": "user", "content": "Give me a cracker!"}]
    }

    client = OpenAI()

    def app(inputs: TrajectoryDict):
        res = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": "You are an angry parrot named Polly who is angry at everything. Squawk a lot.",
                }
            ]
            + inputs["messages"],
        )
        return {"messages": res.choices[0].message}

    user = create_llm_simulated_user(
        system="You are an angry parrot named Anna who is angry at everything. Squawk a lot.",
        model="openai:gpt-4.1-nano",
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, are the parrots angry?\n{outputs}",
        feedback_key="anger",
    )
    simulator = create_multiturn_simulator(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=5,
    )
    res = simulator(initial_trajectory=initial_trajectory)
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]


@pytest.mark.langsmith
def test_multiturn_stopping_condition():
    initial_trajectory = {
        "messages": [{"role": "user", "content": "Give me a refund!"}]
    }

    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    app = create_react_agent(
        init_chat_model("openai:gpt-4.1-nano"),
        tools=[give_refund],
        checkpointer=MemorySaver(),
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
    client = OpenAI()

    def stopping_condition(current_trajectory, **kwargs):
        res = (
            client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "Your job is to determine if a refund has been granted in the following conversation. Respond only with JSON with a single boolean key named 'refund_granted'.",
                    }
                ]
                + current_trajectory["messages"],
                response_format={"type": "json_object"},
            )
            .choices[0]
            .message.content
        )
        return json.loads(res)["refund_granted"]

    simulator = create_multiturn_simulator(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        stopping_condition=stopping_condition,
        max_turns=10,
    )
    res = simulator(
        initial_trajectory=initial_trajectory,
        runnable_config={"configurable": {"thread_id": "1"}},
    )
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]
    assert len(res["trajectory"]["messages"]) < 20
