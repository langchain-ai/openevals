from evaluators.trajectory.llm import create_trajectory_llm_as_judge, DEFAULT_PROMPT

from evaluators.types import ChatCompletionMessage

import pytest
import json


@pytest.mark.langsmith
def test_trajectory_match():
    evaluator = create_trajectory_llm_as_judge(prompt=DEFAULT_PROMPT)
    inputs = {}
    outputs = [
        {"role": "user", "content": "What is the weather in SF?"},
        {"role": "assistant", "tool_calls": [
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
        {"role": "assistant", "tool_calls": [
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
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs,
    )
    assert eval_result["key"] == "trajectory_accuracy"
    assert eval_result["score"] == 1.0


@pytest.mark.langsmith
def test_trajectory_match_with_inverse_rubric():
    evaluator = create_trajectory_llm_as_judge(prompt=DEFAULT_PROMPT)
    inputs = {}
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
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs,
        rubric="We are looking for bad trajectories, so score should be 0 if the trajectory contains reasonable steps for the agent to answer the input, and 1 if not.",
    )
    assert eval_result["key"] == "trajectory_accuracy"
    assert eval_result["score"] is False
