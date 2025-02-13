from evaluators.trajectory.unordered import trajectory_unordered_match
from evaluators.trajectory.superset import trajectory_superset
from evaluators.trajectory.subset import trajectory_subset
from evaluators.trajectory.strict import trajectory_strict_match

from evaluators.types import EvaluatorResult, ChatCompletionMessage

import pytest
import json


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "evaluator, feedback_key",
    [
        (trajectory_unordered_match, "trajectory_unordered_match"),
        (trajectory_superset, "trajectory_superset"),
        (trajectory_subset, "trajectory_subset"),
        (trajectory_strict_match, "trajectory_strict_match"),
    ],
)
def test_trajectory_match(evaluator, feedback_key):
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
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=1.0)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "evaluator, feedback_key",
    [
        (trajectory_unordered_match, "trajectory_unordered_match"),
        (trajectory_superset, "trajectory_superset"),
        (trajectory_subset, "trajectory_subset"),
        (trajectory_strict_match, "trajectory_strict_match"),
    ],
)
def test_trajectory_with_different_tool_message_order(evaluator, feedback_key):
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
    ) == EvaluatorResult(key=feedback_key, score=1.0)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "evaluator, feedback_key, score",
    [
        (trajectory_unordered_match, "trajectory_unordered_match", 1.0),
        (trajectory_superset, "trajectory_superset", 1.0),
        (trajectory_subset, "trajectory_subset", 1.0),
        (trajectory_strict_match, "trajectory_strict_match", 0.0),
    ],
)
def test_trajectory_with_different_message_count(evaluator, feedback_key, score):
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
    ) == EvaluatorResult(key=feedback_key, score=score)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "evaluator, feedback_key, score",
    [
        (trajectory_unordered_match, "trajectory_unordered_match", 0.0),
        (trajectory_superset, "trajectory_superset", 0.0),
        (trajectory_subset, "trajectory_subset", 1.0),
        (trajectory_strict_match, "trajectory_strict_match", 0.0),
    ],
)
def test_trajectory_subset_tool_call(evaluator, feedback_key, score):
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
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "evaluator, feedback_key",
    [
        (trajectory_unordered_match, "trajectory_unordered_match"),
        (trajectory_superset, "trajectory_superset"),
        (trajectory_subset, "trajectory_subset"),
        (trajectory_strict_match, "trajectory_strict_match"),
    ],
)
def test_exact_matcher_with_different_called_tools(evaluator, feedback_key):
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
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=0.0)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "evaluator, feedback_key, score",
    [
        (trajectory_unordered_match, "trajectory_unordered_match", 0.0),
        (trajectory_superset, "trajectory_superset", 1.0),
        (trajectory_subset, "trajectory_subset", 0.0),
        (trajectory_strict_match, "trajectory_strict_match", 0.0),
    ],
)
def test_trajectory_with_extra_tool_calls(evaluator, feedback_key, score):
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
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "evaluator, feedback_key, score",
    [
        (trajectory_unordered_match, "trajectory_unordered_match", 0.0),
        (trajectory_superset, "trajectory_superset", 0.0),
        (trajectory_subset, "trajectory_subset", 1.0),
        (trajectory_strict_match, "trajectory_strict_match", 0.0),
    ],
)
def test_trajectory_with_subset_tool_calls(evaluator, feedback_key, score):
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
    assert evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    ) == EvaluatorResult(key=feedback_key, score=score)
