import json
import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from openevals.llm import create_llm_as_judge

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain import hub as prompts
from langchain_core.messages import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate


@pytest.mark.langsmith
def test_prompt_hub_works():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt=prompts.pull("langchain-ai/test-equality"),
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None
    assert eval_result["comment"] is not None


@pytest.mark.langsmith
def test_prompt_hub_works_one_message():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt=prompts.pull("langchain-ai/equality-1-message"),
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None
    assert eval_result["comment"] is not None


@pytest.mark.langsmith
def test_structured_prompt():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = Client()
    prompt = client.pull_prompt("jacob/simple-equality-structured")
    llm_as_judge = create_llm_as_judge(
        prompt=prompt,
        model="openai:gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["equality"] is True
    assert eval_result["justification"] is not None


@pytest.mark.langsmith
def test_llm_as_judge_openai():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None
    assert eval_result["comment"] is not None


@pytest.mark.langsmith
def test_llm_as_judge_openai_no_reasoning():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
        model="gpt-4o-mini",
        use_reasoning=False,
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None
    assert eval_result["comment"] is None


@pytest.mark.langsmith
def test_llm_as_judge_openai_not_equal():
    inputs = {"a": 1, "b": 3}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_openai_not_equal_continuous():
    inputs = {"a": 1, "b": 3}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt="How equal are these 2? Your score should be a fraction of how many props are equal: {inputs} {outputs}",
        judge=client,
        model="gpt-4o-mini",
        continuous=True,
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] > 0
    assert eval_result["score"] < 1


@pytest.mark.langsmith
def test_llm_as_judge_openai_not_equal_binary_fail():
    inputs = {"a": 1, "b": 3}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_openai_not_equal_binary_pass():
    inputs = {"a": 1, "b": 3}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt="How equal are these 2? Your score should be a fraction of how many props are equal: {inputs} {outputs}",
        judge=client,
        model="o3-mini",
        continuous=True,
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] > 0


@pytest.mark.langsmith
def test_llm_as_judge_langchain():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = ChatOpenAI(model="gpt-4o-mini")
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_langchain_messages():
    inputs = [HumanMessage(content=json.dumps({"a": 1, "b": 2}))]
    outputs = [HumanMessage(content=json.dumps({"a": 1, "b": 3}))]
    client = ChatOpenAI(model="gpt-4o-mini")
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_langchain_messages_dict():
    inputs = {"messages": [HumanMessage(content=json.dumps({"a": 1, "b": 2}))]}
    outputs = {"messages": [HumanMessage(content=json.dumps({"a": 1, "b": 3}))]}
    client = ChatOpenAI(model="gpt-4o-mini")
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        judge=client,
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_init_chat_model():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        model="openai:gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_few_shot_examples():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two foo? {inputs} {outputs}",
        few_shot_examples=[
            {"inputs": {"a": 1, "b": 2}, "outputs": {"a": 1, "b": 2}, "score": 0.0},
            {"inputs": {"a": 1, "b": 3}, "outputs": {"a": 1, "b": 2}, "score": 1.0},
        ],
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert not eval_result["score"]


@pytest.mark.langsmith
def test_llm_as_judge_custom_output_schema_typed_dict():
    class EqualityResult(TypedDict):
        justification: str
        are_equal: bool

    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        output_schema=EqualityResult,
        model="openai:gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["are_equal"]
    assert eval_result["justification"] is not None


@pytest.mark.langsmith
def test_llm_as_judge_custom_output_schema_openai_client():
    class EqualityResult(BaseModel):
        justification: str
        are_equal: bool

    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        output_schema=EqualityResult.model_json_schema(),
        judge=client,
        model="gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["are_equal"]
    assert eval_result["justification"] is not None


@pytest.mark.langsmith
def test_llm_as_judge_custom_output_schema_pydantic():
    class EqualityResult(BaseModel):
        justification: str
        are_equal: bool

    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        output_schema=EqualityResult,
        model="openai:gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert isinstance(eval_result, EqualityResult)
    assert eval_result.are_equal
    assert eval_result.justification is not None


@pytest.mark.langsmith
def test_llm_as_judge_with_evaluate():
    client = Client()
    evaluator = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        model="openai:gpt-4o-mini",
    )
    res = client.evaluate(lambda x: x, data="exact match", evaluators=[evaluator])
    for r in res:
        assert r["evaluation_results"]["results"][0].score is not None


@pytest.mark.langsmith
def test_llm_as_judge_mustache_prompt():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    prompt = ChatPromptTemplate(
        [
            ("system", "You are an expert at determining if two objects are equal."),
            ("human", "Are these two equal? {{inputs}} {{outputs}}"),
        ],
        template_format="mustache",
    )
    llm_as_judge = create_llm_as_judge(prompt=prompt, model="openai:gpt-4o-mini")
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"]
