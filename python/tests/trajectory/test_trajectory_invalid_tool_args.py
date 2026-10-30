import json

import pytest

from openevals.trajectory.match import (
    create_async_trajectory_match_evaluator,
    create_trajectory_match_evaluator,
)

MODES = ["strict", "unordered", "subset", "superset"]


def _trajectory(arguments: str) -> list[dict]:
    return [
        {"role": "user", "content": "What time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "1",
                    "function": {"name": "get_flight_info", "arguments": arguments},
                }
            ],
        },
    ]


REFERENCE = _trajectory(json.dumps({"flight": "AA1"}))


@pytest.mark.parametrize("trajectory_match_mode", MODES)
@pytest.mark.parametrize("arguments", ['{"flight": "AA1"', "[1]", "null"])
@pytest.mark.parametrize("tool_args_match_mode", ["exact", "subset", "superset"])
def test_invalid_args_fail_the_match(
    trajectory_match_mode, arguments, tool_args_match_mode
):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        tool_args_match_mode=tool_args_match_mode,
    )
    result = evaluator(outputs=_trajectory(arguments), reference_outputs=REFERENCE)
    assert result["score"] is False


@pytest.mark.parametrize("trajectory_match_mode", MODES)
def test_invalid_args_match_when_args_are_ignored(trajectory_match_mode):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        tool_args_match_mode="ignore",
    )
    result = evaluator(
        outputs=_trajectory('{"flight": "AA1"'), reference_outputs=REFERENCE
    )
    assert result["score"] is True


@pytest.mark.parametrize("trajectory_match_mode", MODES)
@pytest.mark.parametrize("arguments", ["", "  "])
def test_blank_args_are_empty_args(trajectory_match_mode, arguments):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
    )
    result = evaluator(
        outputs=_trajectory(arguments), reference_outputs=_trajectory("{}")
    )
    assert result["score"] is True


@pytest.mark.parametrize("trajectory_match_mode", MODES)
@pytest.mark.asyncio
async def test_invalid_args_fail_the_match_async(trajectory_match_mode):
    evaluator = create_async_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
    )
    result = await evaluator(
        outputs=_trajectory('{"flight": "AA1"'), reference_outputs=REFERENCE
    )
    assert result["score"] is False
