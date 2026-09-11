from openevals.trajectory.match import create_async_trajectory_match_evaluator

from openevals.types import EvaluatorResult, ChatCompletionMessage

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import json
import pytest


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feedback_key, match_mode",
    [
        ("trajectory_unordered_match", "unordered"),
        ("trajectory_superset_match", "superset"),
        ("trajectory_subset_match", "subset"),
        ("trajectory_strict_match", "strict"),
    ],
)
async def test_trajectory_match(feedback_key, match_mode):
    evaluator = create_async_trajectory_match_evaluator(
        trajectory_match_mode=match_mode
    )
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
    assert await evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(
        key=feedback_key,
        score=True,
        comment=None,
        metadata=None,
    )


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feedback_key, match_mode, score",
    [
        ("trajectory_unordered_match", "unordered", True),
        ("trajectory_superset_match", "superset", True),
        ("trajectory_subset_match", "subset", True),
        ("trajectory_strict_match", "strict", False),
    ],
)
async def test_trajectory_with_different_message_count(feedback_key, match_mode, score):
    evaluator = create_async_trajectory_match_evaluator(
        trajectory_match_mode=match_mode
    )
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
    assert await evaluator(
        outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None, metadata=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_args_match_mode, score",
    [
        ("exact", False),
        ("ignore", True),
    ],
)
async def test_trajectory_match_strict_params(tool_args_match_mode, score):
    evaluator = create_async_trajectory_match_evaluator(
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
            tool_calls=[
                {"id": "1234", "name": "get_weather", "args": {"city": "San Francisco"}}
            ],
        ),
        ToolMessage(tool_call_id="1234", content="It's 80 degrees and sunny in SF."),
        AIMessage("The weather in SF is 80 degrees and sunny."),
    ]
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["key"] == "trajectory_strict_match"
    assert result["score"] == score


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_trajectory_match_with_nested_field_overrides():
    outputs = [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps(
                            {
                                "query": "flight upgrades",
                                "time": {
                                    "start": "2025-03-22T18:34:40Z",
                                    "end": "2025-03-22T20:34:40Z",
                                },
                            }
                        ),
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
                        "arguments": json.dumps(
                            {
                                "query": "foo",
                                "time": {"start": "2025-03-22T18:34:40Z", "end": "baz"},
                            }
                        ),
                    }
                }
            ],
        },
        {"role": "assistant", "content": "Upgrades possible."},
    ]
    evaluator = create_async_trajectory_match_evaluator(
        trajectory_match_mode="strict",
        tool_args_match_overrides={"lookup_policy": ["time.start"]},
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["score"] is True


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_trajectory_match_with_nested_field_overrides_requires_key():
    outputs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps(
                            {"time": {"end": "2025-03-22T20:34:40Z"}}
                        ),
                    }
                }
            ],
        },
    ]
    reference_outputs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps(
                            {"time": {"end": "2025-03-22T20:34:40Z"}}
                        ),
                    }
                }
            ],
        },
    ]
    evaluator = create_async_trajectory_match_evaluator(
        trajectory_match_mode="strict",
        tool_args_match_overrides={"lookup_policy": ["time.start"]},
    )
    result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["score"] is False


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize("match_mode", ["unordered", "superset", "subset", "strict"])
async def test_trajectory_match_rejects_missing_outputs(match_mode):
    """A missing trajectory is unevaluable and must raise, not score.

    Previously `None` was normalized to `[]` before reaching the scorer's own
    `outputs is None` guard, so the guard could never fire. In `subset` mode the
    empty list was then vacuously a subset and the evaluator returned True -- a
    pass for a trajectory that was never observed.
    """
    evaluator = create_async_trajectory_match_evaluator(
        trajectory_match_mode=match_mode
    )
    reference_outputs = [
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
    ]
    with pytest.raises(ValueError):
        await evaluator(outputs=None, reference_outputs=reference_outputs)
    with pytest.raises(ValueError):
        await evaluator(outputs=reference_outputs, reference_outputs=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_trajectory_match_still_accepts_empty_outputs():
    """An empty list is supplied-and-empty, not missing, and is still scored.

    This pins the scope of the `None` change above: only a missing trajectory is
    unevaluable. `[]` remains vacuously a subset, as before.
    """
    evaluator = create_async_trajectory_match_evaluator(trajectory_match_mode="subset")
    reference_outputs = [
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
    ]
    result = await evaluator(outputs=[], reference_outputs=reference_outputs)
    assert result["score"] is True
