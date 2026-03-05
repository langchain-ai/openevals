from openevals.trajectory.match import create_trajectory_match_evaluator

from openevals.types import EvaluatorResult, ChatCompletionMessage

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import json
import pytest


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("trajectory_unordered_match", "unordered"),
        ("trajectory_superset_match", "superset"),
        ("trajectory_subset_match", "subset"),
        ("trajectory_strict_match", "strict"),
    ],
)
def test_trajectory_match(feedback_key, match_mode):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    inputs = {}
    outputs = [
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
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(
        key=feedback_key,
        score=True,
        comment=None,
        metadata=None,
    )


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("trajectory_unordered_match", "unordered"),
        ("trajectory_superset_match", "superset"),
        ("trajectory_subset_match", "subset"),
        ("trajectory_strict_match", "strict"),
    ],
)
def test_trajectory_with_different_tool_message_order(feedback_key, match_mode):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    inputs = {}
    outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        ),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(
        key=feedback_key,
        score=True,
        comment=None,
        metadata=None,
    )


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("trajectory_unordered_match", "unordered", True),
        ("trajectory_superset_match", "superset", True),
        ("trajectory_subset_match", "subset", True),
        ("trajectory_strict_match", "strict", False),
    ],
)
def test_trajectory_with_different_message_count(feedback_key, match_mode, score):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    inputs = {}
    outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        ),
    ]
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("trajectory_unordered_match", "unordered", False),
        ("trajectory_superset_match", "superset", False),
        ("trajectory_subset_match", "subset", True),
        ("trajectory_strict_match", "strict", False),
    ],
)
def test_trajectory_subset_tool_call(feedback_key, match_mode, score):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(role="tool", content="It's 80 degrees and sunny in SF."),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80 degrees and sunny. In London, it's 9000 degrees and hallucinating.",
        ),
    ]
    reference_outputs = [
        ChatCompletionMessage(
            role="user", content="What is the weather in SF and London?"
        ),
        ChatCompletionMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "London"}),
                    }
                },
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    }
                },
            ],
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        ),
    ]
    assert evaluator(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("trajectory_unordered_match", "unordered"),
        ("trajectory_superset_match", "superset"),
        ("trajectory_subset_match", "subset"),
        ("trajectory_strict_match", "strict"),
    ],
)
def test_trajectory_match_langchain_messages(feedback_key, match_mode):
    evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
    outputs = [
        HumanMessage("What is the weather in SF?"),
        AIMessage(
            content="",
            tool_calls=[{"id": "1234", "name": "get_weather", "args": {"city": "San Francisco"}}],
        ),
        ToolMessage(tool_call_id="1234", content="It's 80 degrees and sunny in SF."),
        AIMessage("The weather in SF is 80 degrees and sunny."),
    ]
    reference_outputs = [
        HumanMessage("What is the weather in SF?"),
        AIMessage(
            content="Let me check that for you!",
            tool_calls=[{"id": "4321", "name": "get_weather", "args": {"city": "San Francisco"}}],
        ),
        ToolMessage(tool_call_id="4321", content="It's 80 degrees and sunny in San Francisco."),
        AIMessage("The weather in SF is 80˚ and sunny."),
    ]
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == feedback_key
    assert result["score"] is True


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "tool_args_match_mode, score",
    [
        ("exact", False),
        ("ignore", True),
    ],
)
def test_trajectory_match_strict_params(tool_args_match_mode, score):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="strict",
        tool_args_match_mode=tool_args_match_mode,
    )
    outputs = [
        HumanMessage("What is the weather in SF?"),
        AIMessage(
            content="",
            tool_calls=[{"id": "1234", "name": "get_weather", "args": {"city": "SF"}}],
        ),
        ToolMessage(tool_call_id="1234", content="It's 80 degrees and sunny in SF."),
        AIMessage("The weather in SF is 80 degrees and sunny."),
    ]
    reference_outputs = [
        HumanMessage("What is the weather in SF?"),
        AIMessage(
            content="",
            tool_calls=[{"id": "1234", "name": "get_weather", "args": {"city": "San Francisco"}}],
        ),
        ToolMessage(tool_call_id="1234", content="It's 80 degrees and sunny in SF."),
        AIMessage("The weather in SF is 80 degrees and sunny."),
    ]
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "trajectory_strict_match"
    assert result["score"] == score


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "tool_args_match_mode, score",
    [
        ("exact", False),
        ("ignore", True),
        ("subset", False),
        ("superset", True),
    ],
)
def test_tool_args_match_mode_superset(tool_args_match_mode, score):
    evaluator = create_trajectory_match_evaluator(
        tool_args_match_mode=tool_args_match_mode
    )
    outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_flight_info",
                        "arguments": json.dumps({"is_cool": True, "flight_no": "LX0112"}),
                    }
                }
            ],
        },
        {"role": "assistant", "content": "Your flight is at 10:00 AM."},
    ]
    reference_outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_flight_info",
                        "arguments": json.dumps({"flight_no": "LX0112"}),
                    }
                }
            ],
        },
        {"role": "assistant", "content": "Your flight is at 10:00 AM."},
    ]
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["score"] == score


@pytest.mark.langsmith
def test_trajectory_match_with_nested_field_overrides():
    outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps({
                            "query": "flight upgrades",
                            "time": {"start": "2025-03-22T18:34:40Z", "end": "2025-03-22T20:34:40Z"},
                        }),
                    }
                }
            ],
        },
        {"role": "assistant", "content": "No upgrades available."},
    ]
    reference_outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps({
                            "query": "foo",
                            "time": {"start": "2025-03-22T18:34:40Z", "end": "baz"},
                        }),
                    }
                }
            ],
        },
        {"role": "assistant", "content": "Upgrades possible."},
    ]
    evaluator_no_overrides = create_trajectory_match_evaluator(
        trajectory_match_mode="strict"
    )
    result_no_overrides = evaluator_no_overrides(
        outputs=outputs, reference_outputs=reference_outputs
    )
    assert result_no_overrides["score"] is False

    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="strict",
        tool_args_match_overrides={"lookup_policy": ["time.start"]},
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["score"] is True
