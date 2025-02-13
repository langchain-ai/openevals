from evaluators.types import ChatCompletionMessage, EvaluatorResult
from evaluators.utils import _run_evaluator

from typing import Any


def exact_trajectory_match(
    *,
    outputs: list[ChatCompletionMessage],
    reference_outputs: list[ChatCompletionMessage],
    **kwargs: Any,
) -> EvaluatorResult:
    """
    Assert agent trajectory and called tools matches output agent trajectory and called tools for some given input.

    Args:
        outputs (list[ChatCompletionMessage]): Actual trajectory the agent followed
        reference_outputs (list[ChatCompletionMessage]): Ideal reference trajectory the agent should have followed

    Returns:
        MatchResult: Contains match result with score 1.0 if trajectory (including called tools) matches, 0.0 otherwise
    """

    def get_score():
        if outputs is None or reference_outputs is None:
            raise ValueError(
                "Exact trajectory match requires both outputs and reference_outputs"
            )
        exact_match = True
        for output, reference_output in zip(outputs, reference_outputs):
            if output["role"] != reference_output["role"]:
                exact_match = False
                break
            elif "tool_calls" in output and "tool_calls" in reference_output:
                # Handle case where one has tool calls and other doesn't
                if (
                    output["tool_calls"] is None
                    or reference_output["tool_calls"] is None
                ):
                    exact_match = False
                    break
                output_tool_counts = {}
                reference_tool_counts = {}

                for call in output["tool_calls"]:
                    name = call["function"]["name"]
                    output_tool_counts[name] = output_tool_counts.get(name, 0) + 1

                for call in reference_output["tool_calls"]:
                    name = call["function"]["name"]
                    reference_tool_counts[name] = reference_tool_counts.get(name, 0) + 1

                # Check if tools are called the same number of times
                if output_tool_counts != reference_tool_counts:
                    exact_match = False
                    break
        return 1.0 if exact_match else 0.0

    return _run_evaluator(
        run_name="exact_trajectory_match",
        evaluator_fn=get_score,
        feedback_key="trajectory_match",
    )
