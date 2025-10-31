import json
import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from openevals.llm import create_async_llm_as_judge

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.structured import StructuredPrompt
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain import hub as prompts


@pytest.fixture(scope="session", autouse=True)
def setup_prompts():
    """Setup required prompts in LangChain Hub before running tests."""
    client = Client()

    # Create test-equality prompt
    test_equality_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert LLM as judge.",
            ),
            (
                "human",
                "Are these two equal? {inputs} {outputs}",
            ),
        ]
    )

    try:
        client.push_prompt("test-equality", object=test_equality_prompt)
        print("Created test-equality prompt")
    except Exception as e:
        print(f"test-equality prompt may already exist: {e}")

    # Create equality-1-message prompt
    equality_1_message_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "Are these two equal? {inputs} {outputs}",
            )
        ]
    )

    try:
        client.push_prompt("equality-1-message", object=equality_1_message_prompt)
        print("Created equality-1-message prompt")
    except Exception as e:
        print(f"equality-1-message prompt may already exist: {e}")

    # Create simple-equality-structured prompt
    structured_equality_prompt = StructuredPrompt(
        messages=[
            (
                "human",
                """
Are these equal?

<item1>
{inputs}
</item1>

<item2>
{outputs}
</item2>
""",
            ),
        ],
        schema={
            "title": "score",
            "description": "Get a score",
            "type": "object",
            "properties": {
                "equality": {
                    "type": "boolean",
                    "description": "Whether the two items are equal",
                },
                "justification": {
                    "type": "string",
                    "description": "Justification for your decision above",
                },
            },
            "required": ["equality", "justification"],
            "strict": True,
            "additionalProperties": False,
        },
    )

    try:
        client.push_prompt(
            "simple-equality-structured", object=structured_equality_prompt
        )
        print("Created simple-equality-structured prompt")
    except Exception as e:
        print(f"simple-equality-structured prompt may already exist: {e}")

    return True


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_prompt_hub_works():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = AsyncOpenAI()
    llm_as_judge = create_async_llm_as_judge(
        prompt=prompts.pull("test-equality"),
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None
    assert eval_result["comment"] is not None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_prompt_hub_works_one_message():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = AsyncOpenAI()
    llm_as_judge = create_async_llm_as_judge(
        prompt=prompts.pull("equality-1-message"),
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None
    assert eval_result["comment"] is not None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_structured_prompt():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = Client()
    prompt = client.pull_prompt("simple-equality-structured")
    llm_as_judge = create_async_llm_as_judge(
        prompt=prompt,
        model="openai:gpt-4o-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["equality"] is True
    assert eval_result["justification"] is not None


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
        prompt="How equal are these 2? Your score should be a fraction of how many props are equal: {inputs} {outputs}",
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
        model="o4-mini",
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
        prompt="How equal are these 2? Your score should be a fraction of how many props are equal: {inputs} {outputs}",
        judge=client,
        model="o4-mini",
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
async def test_llm_as_judge_langchain_messages():
    inputs = [HumanMessage(content=json.dumps({"a": 1, "b": 2}))]
    outputs = [HumanMessage(content=json.dumps({"a": 1, "b": 3}))]
    client = ChatOpenAI(model="gpt-4o-mini")
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_llm_as_judge_langchain_messages_dict():
    inputs = {"messages": [HumanMessage(content=json.dumps({"a": 1, "b": 2}))]}
    outputs = {"messages": [HumanMessage(content=json.dumps({"a": 1, "b": 3}))]}
    client = ChatOpenAI(model="gpt-4o-mini")
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_init_chat_model():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        model="openai:o4-mini",
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
        model="openai:o4-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_custom_output_schema_typed_dict():
    class EqualityResult(TypedDict):
        justification: str
        are_equal: bool

    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        output_schema=EqualityResult,
        model="openai:gpt-4o-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["are_equal"]
    assert eval_result["justification"] is not None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_custom_output_schema_openai_client():
    class EqualityResult(BaseModel):
        justification: str
        are_equal: bool

    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = AsyncOpenAI()
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        output_schema=EqualityResult.model_json_schema(),
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["are_equal"]
    assert eval_result["justification"] is not None


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_async_llm_as_judge_custom_output_schema_pydantic():
    class EqualityResult(BaseModel):
        justification: str
        are_equal: bool

    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    llm_as_judge = create_async_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        output_schema=EqualityResult,
        model="openai:gpt-4o-mini",
    )
    eval_result = await llm_as_judge(inputs=inputs, outputs=outputs)
    assert isinstance(eval_result, EqualityResult)
    assert eval_result.are_equal
    assert eval_result.justification is not None
