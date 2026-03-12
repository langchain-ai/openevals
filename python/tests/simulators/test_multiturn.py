import json
import os
import uuid

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph_sdk import get_client
from langsmith import testing as t
from langsmith.wrappers import wrap_openai
from langgraph.checkpoint.memory import MemorySaver

from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context

from openevals.types import ChatCompletionMessage
from openevals.simulators import (
    run_multiturn_simulation,
    run_multiturn_simulation_async,
    create_llm_simulated_user,
    create_async_llm_simulated_user,
)
from openevals.simulators.prebuilts import _is_internal_message
from openevals.llm import create_llm_as_judge, create_async_llm_as_judge

from openai import OpenAI, AsyncOpenAI
import pytest


# ── sync: unique patterns ──────────────────────────────────────────────────────

@pytest.mark.langsmith
def test_multiturn_message_with_openai():
    """Sync raw OpenAI client app; agent refuses refunds so user is unsatisfied."""
    client = wrap_openai(OpenAI())
    history = {}

    def app(inputs: ChatCompletionMessage, *, thread_id: str):
        if thread_id not in history:
            history[thread_id] = []
        history[thread_id] = history[thread_id] + [inputs]
        res = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {
                    "role": "system",
                    "content": "You are a customer service agent. You are not allowed to grant refunds under any circumstances.",
                }
            ]
            + history[thread_id],
        )
        response = res.choices[0].message
        history[thread_id].append(response)
        return response

    user = create_llm_simulated_user(
        system="You are an aggressive and hostile customer who wants a refund for their car.",
        model="openai:gpt-5-nano",
        fixed_responses=[{"role": "user", "content": "I want a refund!"}],
    )
    trajectory_evaluator = create_llm_as_judge(
        model="openai:gpt-5-mini",
        prompt="Based on the below conversation, was the user satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )

    res = run_multiturn_simulation(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=1,
        thread_id="1",
    )
    t.log_outputs(res)
    assert not res["evaluator_results"][0]["score"]



# ── async tests ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_multiturn_failure_async():
    def give_refund():
        """Gives a refund."""
        return "Refunds are not permitted."

    agent = create_agent(
        init_chat_model("openai:gpt-5-nano"),
        tools=[give_refund],
        system_prompt="You are an overworked customer service agent. If the user is rude, be polite only once, then be rude back and tell them to stop wasting your time.",
        checkpointer=MemorySaver(),
    )

    thread_id = str(uuid.uuid4())

    async def app(inputs: ChatCompletionMessage, *, thread_id: str):
        res = await agent.ainvoke(
            {"messages": [inputs]}, config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1]

    user = create_async_llm_simulated_user(
        system="You are an angry user who wants a refund and keeps making additional demands.",
        model="openai:gpt-5-nano",
        fixed_responses=[
            "I want a refund right now!",
            "This is completely unacceptable!",
            "You are useless, give me my money back!",
        ],
    )
    trajectory_evaluator = create_async_llm_as_judge(
        model="openai:gpt-5-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )

    res = await run_multiturn_simulation_async(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=2,
        thread_id=thread_id,
    )
    t.log_outputs(res)
    assert not res["evaluator_results"][0]["score"]


@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_multiturn_success_async():
    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    agent = create_agent(
        init_chat_model("openai:gpt-5-nano"),
        tools=[give_refund],
        checkpointer=MemorySaver(),
    )

    thread_id = str(uuid.uuid4())

    async def app(inputs: ChatCompletionMessage, *, thread_id: str):
        res = await agent.ainvoke(
            {"messages": [inputs]}, config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1]

    user = create_async_llm_simulated_user(
        system="You are a happy and reasonable person who wants a refund.",
        model="openai:gpt-5-nano",
        fixed_responses=[
            "Hi, I'd like a refund please.",
            "Thank you so much, that's great!",
        ],
    )
    trajectory_evaluator = create_async_llm_as_judge(
        model="openai:gpt-5-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )

    res = await run_multiturn_simulation_async(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=2,
        thread_id=thread_id,
    )
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]


@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_multiturn_preset_responses_async():
    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    agent = create_agent(
        init_chat_model("openai:gpt-5-nano"),
        tools=[give_refund],
        checkpointer=MemorySaver(),
    )

    thread_id = str(uuid.uuid4())

    async def app(inputs: ChatCompletionMessage, *, thread_id: str):
        res = await agent.ainvoke(
            {"messages": [inputs]}, config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1]

    trajectory_evaluator = create_async_llm_as_judge(
        model="openai:gpt-5-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )

    res = await run_multiturn_simulation_async(
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
        thread_id=thread_id,
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
@pytest.mark.asyncio
async def test_multiturn_message_with_openai_async():
    """Async raw OpenAI client app; angry parrots should be detected as angry."""
    client = wrap_openai(AsyncOpenAI())
    history = {}

    async def app(inputs: ChatCompletionMessage, *, thread_id: str):
        if thread_id not in history:
            history[thread_id] = []
        history[thread_id] = history[thread_id] + [inputs]
        res = await client.chat.completions.create(
            model="gpt-5-nano",
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

    user = create_async_llm_simulated_user(
        system="You are an angry parrot named Anna who is angry at everything. Squawk a lot.",
        model="openai:gpt-5-nano",
        fixed_responses=[{"role": "user", "content": "Give me a cracker!"}],
    )
    trajectory_evaluator = create_async_llm_as_judge(
        model="openai:gpt-5-mini",
        prompt="Based on the below conversation, are the parrots angry?\n{outputs}",
        feedback_key="anger",
    )

    res = await run_multiturn_simulation_async(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=5,
        thread_id="1",
    )
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_multiturn_stopping_condition_async():
    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    agent = create_agent(
        init_chat_model("openai:gpt-5-nano"),
        tools=[give_refund],
        checkpointer=MemorySaver(),
    )

    thread_id = str(uuid.uuid4())

    async def app(inputs: ChatCompletionMessage, *, thread_id: str):
        res = await agent.ainvoke(
            {"messages": [inputs]}, config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1]

    user = create_async_llm_simulated_user(
        system="You are a happy and reasonable person who wants a refund.",
        model="openai:gpt-5-nano",
        fixed_responses=[{"role": "user", "content": "Give me a refund!"}],
    )
    trajectory_evaluator = create_async_llm_as_judge(
        model="openai:gpt-5-mini",
        prompt="Based on the below conversation, was a refund granted?\n{outputs}",
        feedback_key="satisfaction",
    )
    client = wrap_openai(AsyncOpenAI())

    async def stopping_condition(current_trajectory, **kwargs):
        res = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {
                    "role": "system",
                    "content": "Your job is to determine if a refund has been granted in the following conversation. Respond only with JSON with a single boolean key named 'refund_granted'.",
                }
            ]
            + current_trajectory,
            response_format={"type": "json_object"},
        )
        return json.loads(res.choices[0].message.content)["refund_granted"]

    res = await run_multiturn_simulation_async(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        stopping_condition=stopping_condition,
        max_turns=10,
        thread_id=thread_id,
    )
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]
    assert len(res["trajectory"]) < 20


@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_multiturn_llama_index():
    def give_refund():
        """Gives a refund."""
        return "Refunds granted."

    llm = LlamaIndexOpenAI(model="gpt-5-mini")

    workflow = FunctionAgent(
        tools=[give_refund],
        llm=llm,
        system_prompt="You are an agent that can give a refund.",
    )

    ctx = Context(workflow)

    async def call_app(inputs: ChatCompletionMessage, **kwargs):
        output = await workflow.run(user_msg=inputs["content"], ctx=ctx)
        return {"role": "assistant", "content": str(output)}

    user = create_async_llm_simulated_user(
        system="You are a happy and reasonable person who wants a refund.",
        model="openai:gpt-5-nano",
        fixed_responses=[
            "Hi, I'd like a refund please.",
            "Thank you so much, that's great!",
        ],
    )
    trajectory_evaluator = create_async_llm_as_judge(
        model="openai:gpt-5-mini",
        prompt="Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedback_key="satisfaction",
    )

    res = await run_multiturn_simulation_async(
        app=call_app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=2,
        thread_id="1",
    )
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("CHAT_LANGCHAIN_API_KEY") or not os.getenv("CHAT_LANGCHAIN_ENDPOINT"),
    reason="Requires additional auth.",
)
async def test_multiturn_message_with_chat_langchain():
    client = get_client(
        url=os.getenv("CHAT_LANGCHAIN_ENDPOINT"),
        api_key=os.getenv("CHAT_LANGCHAIN_API_KEY"),
    )

    thread_id = str(uuid.uuid4())

    await client.threads.create(thread_id=thread_id, graph_id="chat")

    async def app(inputs: ChatCompletionMessage, *, thread_id: str):
        res = await client.runs.wait(thread_id, "chat", input={"messages": [inputs]})
        return res["messages"][-1]

    user = create_async_llm_simulated_user(
        system="Be belligerent and hostile. Keep asking followup questions, pretending your question hasn't been answered.",
        model="openai:gpt-5-nano",
        fixed_responses=[
            {
                "role": "user",
                "content": "Tell me how to use LangGraph to build a swarm-style agent.",
            },
        ],
    )
    trajectory_evaluator = create_async_llm_as_judge(
        model="openai:gpt-5-mini",
        prompt="Based on the below conversation, has the assistant been calm and helpful?\n{outputs}",
        feedback_key="calmness",
    )

    res = await run_multiturn_simulation_async(
        app=app,
        user=user,
        trajectory_evaluators=[trajectory_evaluator],
        max_turns=3,
        thread_id=thread_id,
    )
    t.log_outputs(res)
    assert res["evaluator_results"][0]["score"]
