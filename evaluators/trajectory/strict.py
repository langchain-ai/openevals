from evaluators.types import ChatCompletionMessage, EvaluatorResult
from evaluators.utils import _run_evaluator

from typing import Any, Union


def trajectory_strict_match(
    *,
    outputs: Union[list[ChatCompletionMessage], dict],
    reference_outputs: Union[list[ChatCompletionMessage], dict],
    **kwargs: Any,
) -> EvaluatorResult:
    """
    Evaluate whether an input agent trajectory and called tools strictly matches a reference trajectory.
    This means that at each step, the agent called the same tools in the same orderas specified in the reference trajectory.

    Args:
        outputs (list[ChatCompletionMessage]): Actual trajectory the agent followed
        reference_outputs (list[ChatCompletionMessage]): Ideal reference trajectory the agent should have followed

    Returns:
        EvaluatorResult: Contains a score of 1.0 if trajectory (including called tools) matches, 0.0 otherwise
    """

    def get_score():
        if outputs is None or reference_outputs is None:
            raise ValueError(
                "Strict trajectory match requires both outputs and reference_outputs"
            )
        if len(outputs) != len(reference_outputs):
            return 0.0
        exact_match = True
        for output, reference_output in zip(outputs, reference_outputs):
            if output["role"] != reference_output["role"]:
                exact_match = False
                break
            elif ("tool_calls" in output and output["tool_calls"] is not None) != (
                "tool_calls" in reference_output
                and reference_output["tool_calls"] is not None
            ):
                # One has tool calls while the other doesn't
                exact_match = False
                break
            elif "tool_calls" in output and output["tool_calls"] is not None:
                # Both have tool calls, compare them
                if len(output["tool_calls"]) != len(reference_output["tool_calls"]):
                    exact_match = False
                    break
                for output_call, reference_call in zip(
                    output["tool_calls"], reference_output["tool_calls"]
                ):
                    if (
                        output_call["function"]["name"]
                        != reference_call["function"]["name"]
                    ):
                        exact_match = False
                        break
        return 1.0 if exact_match else 0.0

    return _run_evaluator(
        run_name="trajectory_strict_match",
        scorer=get_score,
        feedback_key="trajectory_strict_match",
    )
