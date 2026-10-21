import json
from copy import deepcopy

import pytest

from openevals.trajectory.match import (
    create_async_trajectory_match_evaluator,
    create_trajectory_match_evaluator,
)
from openevals.trajectory.utils import _normalize_to_openai_messages_list


@pytest.mark.parametrize("container", ["single", "list", "messages"])
@pytest.mark.parametrize(
    "message, expected_field, expected_value",
    [
        ({"role": "assistant", "content": None}, "content", ""),
        ({"role": "tool", "content": "sunny"}, "tool_call_id", ""),
        (
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "weather", "arguments": '{"city":"SF"}'}}
                ],
            },
            "content",
            "",
        ),
    ],
)
def test_normalization_preserves_input_messages(
    container, message, expected_field, expected_value
):
    message = deepcopy(message)
    original = deepcopy(message)
    inputs = message
    if container == "list":
        inputs = [message]
    elif container == "messages":
        inputs = {"messages": [message]}

    normalized = _normalize_to_openai_messages_list(inputs)

    assert message == original
    assert normalized[0][expected_field] == expected_value
    if "tool_calls" in original:
        assert normalized[0]["tool_calls"][0]["id"] == ""
        function = normalized[0]["tool_calls"][0]["function"]
        assert function["name"] == "weather"
        assert json.loads(function["arguments"]) == {"city": "SF"}


@pytest.mark.parametrize("match_mode", ["strict", "unordered", "subset", "superset"])
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_trajectory_evaluator_preserves_input_messages(match_mode, use_async):
    outputs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "weather", "arguments": '{"city":"SF"}'}}
            ],
        },
        {"role": "tool", "content": "sunny"},
    ]
    reference_outputs = deepcopy(outputs)
    original = deepcopy(outputs)

    if use_async:
        evaluator = create_async_trajectory_match_evaluator(
            trajectory_match_mode=match_mode
        )
        result = await evaluator(outputs=outputs, reference_outputs=reference_outputs)
    else:
        evaluator = create_trajectory_match_evaluator(trajectory_match_mode=match_mode)
        result = evaluator(outputs=outputs, reference_outputs=reference_outputs)

    assert result["score"] is True
    assert outputs == original
    assert reference_outputs == original
