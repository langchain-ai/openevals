from evaluators.trajectory.unordered import (
    trajectory_unordered_match_async,
)
from evaluators.trajectory.superset import trajectory_superset_async
from evaluators.trajectory.subset import trajectory_subset_async
from evaluators.trajectory.strict import trajectory_strict_match_async

from evaluators.types import EvaluatorResult, ChatCompletionMessage

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import json
import pytest


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evaluator, feedback_key",
    [
        (trajectory_unordered_match_async, "trajectory_unordered_match"),
        (trajectory_superset_async, "trajectory_superset"),
        (trajectory_subset_async, "trajectory_subset"),
        (trajectory_strict_match_async, "trajectory_strict_match"),
    ],
)
async def test_trajectory_match(evaluator, feedback_key):
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
    assert await evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=True, comment=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evaluator, feedback_key",
    [
        (trajectory_unordered_match_async, "trajectory_unordered_match"),
        (trajectory_superset_async, "trajectory_superset"),
        (trajectory_subset_async, "trajectory_subset"),
        (trajectory_strict_match_async, "trajectory_strict_match"),
    ],
)
async def test_trajectory_with_different_tool_message_order(evaluator, feedback_key):
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
    assert await evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=True, comment=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evaluator, feedback_key, score",
    [
        (trajectory_unordered_match_async, "trajectory_unordered_match", 1.0),
        (trajectory_superset_async, "trajectory_superset", 1.0),
        (trajectory_subset_async, "trajectory_subset", 1.0),
        (trajectory_strict_match_async, "trajectory_strict_match", 0.0),
    ],
)
async def test_trajectory_with_different_message_count(evaluator, feedback_key, score):
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
    assert await evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evaluator, feedback_key, score",
    [
        (trajectory_unordered_match_async, "trajectory_unordered_match", 0.0),
        (trajectory_superset_async, "trajectory_superset", 0.0),
        (trajectory_subset_async, "trajectory_subset", 1.0),
        (trajectory_strict_match_async, "trajectory_strict_match", 0.0),
    ],
)
async def test_trajectory_subset_tool_call(evaluator, feedback_key, score):
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
    assert await evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evaluator, feedback_key",
    [
        (trajectory_unordered_match_async, "trajectory_unordered_match"),
        (trajectory_superset_async, "trajectory_superset"),
        (trajectory_subset_async, "trajectory_subset"),
        (trajectory_strict_match_async, "trajectory_strict_match"),
    ],
)
async def test_exact_matcher_with_different_called_tools(evaluator, feedback_key):
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
                        "name": "accuweather_forecast",
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
    ) == EvaluatorResult(key=feedback_key, score=False, comment=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evaluator, feedback_key, score",
    [
        (trajectory_unordered_match_async, "trajectory_unordered_match", 0.0),
        (trajectory_superset_async, "trajectory_superset", 1.0),
        (trajectory_subset_async, "trajectory_subset", 0.0),
        (trajectory_strict_match_async, "trajectory_strict_match", 0.0),
    ],
)
async def test_trajectory_with_extra_tool_calls(evaluator, feedback_key, score):
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
                        "arguments": json.dumps({"city": "San Francisco"}),
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
        ChatCompletionMessage(
            role="tool", content="It's 80 degrees and sunny in San Francisco."
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80˚ and sunny. In London, it's 90˚ and rainy.",
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
                        "arguments": json.dumps({"city": "SF and London"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(
            role="tool",
            content="It's 80 degrees and sunny in SF, and 90 degrees and rainy in London.",
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        ),
    ]
    assert await evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evaluator, feedback_key, score",
    [
        (trajectory_unordered_match_async, "trajectory_unordered_match", 0.0),
        (trajectory_superset_async, "trajectory_superset", 0.0),
        (trajectory_subset_async, "trajectory_subset", 1.0),
        (trajectory_strict_match_async, "trajectory_strict_match", 0.0),
    ],
)
async def test_trajectory_with_subset_tool_calls(evaluator, feedback_key, score):
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
                        "arguments": json.dumps({"city": "SF and London"}),
                    }
                }
            ],
        ),
        ChatCompletionMessage(
            role="tool",
            content="It's 80 degrees and sunny in SF, and 90 degrees and rainy in London.",
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
                        "arguments": json.dumps({"city": "San Francisco"}),
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
        ChatCompletionMessage(
            role="tool", content="It's 80 degrees and sunny in San Francisco."
        ),
        ChatCompletionMessage(
            role="tool", content="It's 90 degrees and rainy in London."
        ),
        ChatCompletionMessage(
            role="assistant",
            content="The weather in SF is 80˚ and sunny. In London, it's 90˚ and rainy.",
        ),
    ]
    assert await evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score, comment=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evaluator, feedback_key",
    [
        (trajectory_unordered_match_async, "trajectory_unordered_match"),
        (trajectory_superset_async, "trajectory_superset"),
        (trajectory_subset_async, "trajectory_subset"),
        (trajectory_strict_match_async, "trajectory_strict_match"),
    ],
)
async def test_trajectory_match_with_langchain_messages(evaluator, feedback_key):
    inputs = {}
    outputs = [
        HumanMessage(content="What is the weather in SF?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "1234",
                    "name": "get_weather",
                    "args": {"city": "SF"},
                }
            ],
        ),
        ToolMessage(tool_call_id="1234", content="It's 80 degrees and sunny in SF."),
        AIMessage(content="The weather in SF is 80 degrees and sunny."),
    ]
    reference_outputs = [
        HumanMessage(content="What is the weather in SF?"),
        AIMessage(
            content="Let me check that for you!",
            tool_calls=[
                {
                    "id": "4321",
                    "name": "get_weather",
                    "args": {"city": "San Francisco"},
                }
            ],
        ),
        ToolMessage(
            tool_call_id="4321", content="It's 80 degrees and sunny in San Francisco."
        ),
        AIMessage(content="The weather in SF is 80˚ and sunny."),
    ]
    assert await evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=True, comment=None)


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evaluator, feedback_key",
    [
        (trajectory_unordered_match_async, "trajectory_unordered_match"),
        (trajectory_superset_async, "trajectory_superset"),
        (trajectory_subset_async, "trajectory_subset"),
        (trajectory_strict_match_async, "trajectory_strict_match"),
    ],
)
async def test_trajectory_match_with_langchain_messages_failure(
    evaluator, feedback_key
):
    inputs = {}
    outputs = [
        HumanMessage(content="What is the weather in SF?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "1234",
                    "name": "get_weather",
                    "args": {"city": "SF"},
                }
            ],
        ),
        ToolMessage(tool_call_id="1234", content="It's 80 degrees and sunny in SF."),
        AIMessage(content="The weather in SF is 80 degrees and sunny."),
    ]
    reference_outputs = [
        HumanMessage(content="What is the weather in SF?"),
        AIMessage(
            content="Let me check that for you!",
            tool_calls=[
                {
                    "id": "4321",
                    "name": "accuweather_forecast",
                    "args": {"city": "San Francisco"},
                }
            ],
        ),
        ToolMessage(
            tool_call_id="4321", content="It's 80 degrees and sunny in San Francisco."
        ),
        AIMessage(content="The weather in SF is 80˚ and sunny."),
    ]
    assert await evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=False, comment=None)
