from evaluators.llm import create_llm_as_judge
from openai import OpenAI
import pytest
# from langsmith import testing as t


@pytest.mark.langsmith
def test_llm_as_judge_openai():
    inputs = {"a": 1, "b": 2}
    outputs = {"a": 1, "b": 2}
    client = OpenAI()
    llm_as_judge = create_llm_as_judge(
        prompt="Are these two equal? {inputs} {outputs}",
        client=client,
        model="gpt-4o-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] is not None
