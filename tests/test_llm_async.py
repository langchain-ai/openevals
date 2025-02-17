from evaluators.llm import create_async_llm_as_judge
from openai import AsyncOpenAI
import pytest
from langchain_openai import ChatOpenAI


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_openai():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = AsyncOpenAI()
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None
    assert eval_result["comment"] is not None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_openai_no_reasoning():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = AsyncOpenAI()
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
        model="gpt-4o-mini",
        use_reasoning=False,
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None
    assert eval_result["comment"] is None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_openai_not_equal():
    inputs = {"a": 1, "b": 3}
    outputs = {"a": 1, "b": 2}
    client = AsyncOpenAI()
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_openai_not_equal_continuous():
    inputs = {"a": 1, "b": 3}
    outputs = {"a": 1, "b": 2}
    client = AsyncOpenAI()
    llm_as_judge = create_async_llm_as_judge(
        prompt="How equal are these 2? {inputs} {outputs}",
        judge=client,
        model="gpt-4o-mini",
        continuous=True,
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] > 0
    assert eval_result["score"] < 1


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_openai_not_equal_binary_fail():
    inputs = {"a": 1, "b": 3}
    outputs = {"a": 1, "b": 2}
    client = AsyncOpenAI()
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
        model="o3-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_openai_not_equal_binary_pass():
    inputs = {"a": 1, "b": 3}
    outputs = {"a": 1, "b": 2}
    client = AsyncOpenAI()
    llm_as_judge = create_async_llm_as_judge(
        prompt="How equal are these 2? {inputs} {outputs}",
        judge=client,
        model="o3-mini",
        continuous=True,
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] > 0


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_langchain():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = ChatOpenAI(model="gpt-4o-mini")
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_init_chat_model():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        model="openai:gpt-4o-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_few_shot_examples():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two foo? {inputs} {outputs}",
        few_shot_examples=[
            {"inputs": {"a": 1, "b": 2}, "outputs": {"a": 1, "b": 2}, "score": 0.0},
            {"inputs": {"a": 1, "b": 3}, "outputs": {"a": 1, "b": 2}, "score": 1.0},
        ],
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]
