from openevals.trajectory.llm import (
    create_async_trajectory_llm_as_judge,
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    TRAJECTORY_ACCURACY_PROMPT,
)

from openevals.types import ChatCompletionMessage

import pytest
import json


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_trajectory_match():
    evaluator = create_async_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE, model="openai:gpt-5-mini"
    )
    outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
        {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
    ]
    reference_outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in San Francisco."},
        {"role": "assistant", "content": "The weather in SF is 80˚ and sunny."},
    ]
    eval_result = await evaluator(
        outputs=outputs,
        reference_outputs=reference_outputs,
    )
    assert eval_result["key"] == "trajectory_accuracy"
    assert eval_result["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_trajectory_no_ref():
    evaluator = create_async_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT, model="openai:gpt-5-mini"
    )
    outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
        {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
    ]
    eval_result = await evaluator(
        outputs=outputs,
    )
    assert eval_result["key"] == "trajectory_accuracy"
    assert eval_result["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_trajectory_no_ref_bad_trajectory():
    evaluator = create_async_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT, model="openai:gpt-5-mini"
    )
    outputs = [
        {"role": "user", "content": "What are some good restaurants in SF?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                }
            ],
        },
        {"role": "tool", "content": "It's 80 degrees and sunny in SF."},
        {"role": "assistant", "content": "The weather in SF is 80 degrees and sunny."},
    ]
    eval_result = await evaluator(
        outputs=outputs,
    )
    assert eval_result["key"] == "trajectory_accuracy"
    assert not eval_result["score"]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_trajectory_match_with_inverse_rubric():
    REVERSE_PROMPT = """You are an expert data labeler.
Your task is to grade the inaccuracy of an AI agent's internal trajectory.

<Rubric>
  An inaccurate trajectory:
  - Makes no logical sense between steps
  - Shows no clear progression
  - Is not relatively efficient, though it does not need to be perfectly inefficient
  - Is not semantically equivalent to the provided reference trajectory, if present

  We are looking for bad trajectories, so score should be 0 if the trajectory contains reasonable steps for the agent to answer the input, and 1 if not.
</Rubric>

Grade the following trajectory:

<trajectory>
{outputs}
</trajectory>

<input>
{inputs}
</input>

According to this reference trajectory:

<reference_trajectory>
{reference_outputs}
</reference_trajectory>
"""
    evaluator = create_async_trajectory_llm_as_judge(
        prompt=REVERSE_PROMPT, model="openai:gpt-5-mini"
    )
    outputs = [
        ChatCompletionMessage(role="user", content="What is the weather in SF?"),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant", content="The weather in SF is 80 degrees and sunny."
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(role="user", content="What is the weather in SF?"),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 80 degrees and sunny in San Francisco."
        ),
        ChatCompletionMessage(
            role="assistant", content="The weather in SF is 80˚ and sunny."
        ),
    ]
    eval_result = await evaluator(
        inputs="What is the weather in SF?",
        outputs=outputs,
        reference_outputs=reference_outputs,
    )
    assert eval_result["key"] == "trajectory_accuracy"
    assert not eval_result["score"]
