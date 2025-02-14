__all__ = ["_is_trajectory_superset", "_extract_tool_calls"]

from evaluators.types import ChatCompletionMessage


def _normalize_tool_call(tool_call: dict) -> dict:
    return {
        "name": tool_call["function"]["name"],
        "args": tool_call["function"]["arguments"],
    }


def _extract_tool_calls(messages: list[ChatCompletionMessage]) -> list[dict]:
    tool_calls = []
    for message in messages:
        if "tool_calls" in message:
            tool_calls.extend(
                _normalize_tool_call(tool_call) for tool_call in message["tool_calls"]
            )
    return tool_calls


def _is_trajectory_superset(
    outputs: list[ChatCompletionMessage], reference_outputs: list[ChatCompletionMessage]
):
    output_tool_calls = _extract_tool_calls(outputs)
    reference_tool_calls = _extract_tool_calls(reference_outputs)
    output_tool_counts = {}
    reference_tool_counts = {}
    for call in output_tool_calls:
        name = call["name"]
        output_tool_counts[name] = output_tool_counts.get(name, 0) + 1
    for call in reference_tool_calls:
        name = call["name"]
        reference_tool_counts[name] = reference_tool_counts.get(name, 0) + 1
    is_superset = True
    for name in set(output_tool_counts) | set(reference_tool_counts):
        if output_tool_counts.get(name, 0) < reference_tool_counts.get(name, 0):
            is_superset = False
            break
    return is_superset
