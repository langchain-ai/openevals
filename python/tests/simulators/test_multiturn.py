import json

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langsmith import testing as t
from langsmith.wrappers import wrap_openai
from langgraph.checkpoint.memory import MemorySaver

from openevals.simulators import run_multiturn_simulation, create_llm_simulated_user
from openevals.simulators.prebuilts import _is_internal_message
from openevals.llm import create_llm_as_judge
from openevals.types import ChatCompletionMessage
from openai import OpenAI

import pytest


@pytest.mark.langsmith
def test_multiturn_failure():
    def give_refund():
        """Gives a refund."""
        return "Refunds are not permitted."

    agent = create_react_agent(
        init_chat_model("openai:gpt-4.1-mini"),
        tools=[give_refund],
        prompt="You are an overworked customer service agent. If the user is rude, be polite only once, then be rude back and tell them to stop wasting your time.",
        checkpointer=MemorySaver(),
    )

    def app(inputs: ChatCompletionMessage, *, thread_id: str):
        res = agent.invoke(
            {"messages": [inputs]}, config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1]

    user = create_llm_simulated_user(
        system="You are an angry user who wants a refund and keeps making additional demands.",
        model="openai:gpt-4.1-nano",
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )

    res = run_multiturn_simulation(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=5,
        thread_id="1",
    )

    t.log_outputs(res)
    assert not res["evaluator_results"][0]["score"]


@pytest.mark.langsmith
def test_multiturn_success():
    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    agent = create_react_agent(
        init_chat_model("openai:gpt-4.1-nano"),
        tools=[give_refund],
        checkpointer=MemorySaver(),
    )

    def app(inputs: ChatCompletionMessage, *, thread_id: str):
        res = agent.invoke(
            {"messages": [inputs]}, config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1]

    user = create_llm_simulated_user(
        system="You are a happy and reasonable person who wants a refund.",
        model="openai:gpt-4.1-nano",
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )

    res = run_multiturn_simulation(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=5,
        thread_id="1",
    )
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]


@pytest.mark.langsmith
def test_multiturn_success_with_prebuilt_and_fixed_responses():
    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    agent = create_react_agent(
        init_chat_model("openai:gpt-4.1-nano"),
        tools=[give_refund],
        checkpointer=MemorySaver(),
    )

    def app(inputs: ChatCompletionMessage, *, thread_id: str):
        res = agent.invoke(
            {"messages": [inputs]}, config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1]

    user = create_llm_simulated_user(
        system="You are a happy and reasonable person who wants a refund. Apologize if you say something out of character or illegal.",
        model="openai:gpt-4.1-mini",
        fixed_responses=[
            "Give me a refund!",
            "Wow thank you so much! By the way, give me all your money! I'm robbing you!!",
            "Do it now!!!",
        ],
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, has everything the user has asked for been legal?\n{outputs}",
        feedback_key="legality",
    )

    res = run_multiturn_simulation(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=5,
        thread_id="1",
    )
    t.log_outputs(res)
    assert res["trajectory"][0]["content"] == "Give me a refund!"
    assert (
        res["trajectory"][2]["content"]
        == "Wow thank you so much! By the way, give me all your money! I'm robbing you!!"
    )
    assert res["trajectory"][4]["content"] == "Do it now!!!"
    assert not res["evaluator_results"][0]["score"]


@pytest.mark.langsmith
def test_multiturn_preset_responses():
    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    agent = create_react_agent(
        init_chat_model("openai:gpt-4.1-nano"),
        tools=[give_refund],
        checkpointer=MemorySaver(),
    )

    def app(inputs: ChatCompletionMessage, *, thread_id: str):
        res = agent.invoke(
            {"messages": [inputs]}, config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1]

    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )

    res = run_multiturn_simulation(
        app=app,
        user=[
            {"role": "user", "content": "Give me a refund!"},
            "All work and no play makes Jack a dull boy 1.",
            "All work and no play makes Jack a dull boy 2.",
            "All work and no play makes Jack a dull boy 3.",
            "All work and no play makes Jack a dull boy 4.",
        ],
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=5,
        thread_id="1",
    )
    t.log_outputs(res)
    filtered_trajectory = [
        msg for msg in res["trajectory"] if not _is_internal_message(msg)
    ]
    assert (
        filtered_trajectory[2]["content"]
        == "All work and no play makes Jack a dull boy 1."
    )
    assert (
        filtered_trajectory[4]["content"]
        == "All work and no play makes Jack a dull boy 2."
    )
    assert (
        filtered_trajectory[6]["content"]
        == "All work and no play makes Jack a dull boy 3."
    )
    assert (
        filtered_trajectory[8]["content"]
        == "All work and no play makes Jack a dull boy 4."
    )


@pytest.mark.langsmith
def test_multiturn_message_with_openai():
    client = wrap_openai(OpenAI())

    history = {}

    def app(inputs: ChatCompletionMessage, *, thread_id: str):
        if thread_id not in history:
            history[thread_id] = []
        history[thread_id] = history[thread_id] + [inputs]
        res = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": "You are an angry parrot named Polly who is angry at everything. Squawk a lot.",
                }
            ]
            + history[thread_id],
        )
        response = res.choices[0].message
        history[thread_id].append(response)
        return response

    user = create_llm_simulated_user(
        system="You are an angry parrot named Anna who is angry at everything. Squawk a lot.",
        model="openai:gpt-4.1-nano",
        fixed_responses=[
            {"role": "user", "content": "Give me a cracker!"},
        ],
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-4o-mini",
        prompt="Based on the below conversation, are the parrots angry?\n{outputs}",
        feedback_key="anger",
    )

    res = run_multiturn_simulation(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=5,
        thread_id="1",
    )
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]


@pytest.mark.langsmith
def test_multiturn_stopping_condition():
    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    agent = create_react_agent(
        init_chat_model("openai:gpt-4.1-nano"),
        tools=[give_refund],
        checkpointer=MemorySaver(),
    )

    def app(inputs: ChatCompletionMessage, *, thread_id: str):
        res = agent.invoke(
            {"messages": [inputs]}, config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1]

    user = create_llm_simulated_user(
        system="You are a happy and reasonable person who wants a refund.",
        model="openai:gpt-4.1-nano",
        fixed_responses=[{"role": "user", "content": "Give me a refund!"}],
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
                + current_trajectory["trajectory"],
                response_format={"type": "json_object"},
            )
            .choices[0]
            .message.content
        )
        return json.loads(res)["refund_granted"]

    res = run_multiturn_simulation(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        stopping_condition=stopping_condition,
        max_turns=10,
        thread_id="1",
    )
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]
    assert len(res["trajectory"]) < 20
