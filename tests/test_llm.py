from evaluators.llm import create_llm_as_judge
from openai import OpenAI
import pytest
from langchain_openai import ChatOpenAI


@pytest.mark.langsmith
def test_llm_as_judge_openai():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        client_or_scorer=client,
        model="gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None


@pytest.mark.langsmith
def test_llm_as_judge_langchain():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = ChatOpenAI(model="gpt-4o-mini")
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        client_or_scorer=client,
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None


@pytest.mark.langsmith
def test_llm_as_judge_arbitrary_function():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}

    def arbitrary_function(prompt: str) -> float:
        return 1.0

    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        client_or_scorer=arbitrary_function,
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] == 1.0
