import json

import pytest

from openevals.trajectory.match import (
    create_async_trajectory_match_evaluator,
    create_trajectory_match_evaluator,
)


def _trajectory(city: str) -> list[dict]:
    return [
        {"role": "user", "content": "What is the weather in SF?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "1",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": city}),
                    },
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    "trajectory_match_mode", ["strict", "unordered", "subset", "superset"]
)
@pytest.mark.parametrize("override", ["strict", "Exact", "exactly"])
def test_unknown_tool_args_override_mode_raises(trajectory_match_mode, override):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        tool_args_match_overrides={"get_weather": override},
    )
    with pytest.raises(ValueError, match="Invalid tool args match mode"):
        evaluator(
            outputs=_trajectory("NYC"),
            reference_outputs=_trajectory("SF"),
        )


@pytest.mark.parametrize("override, expected", [("exact", False), ("ignore", True)])
def test_known_tool_args_override_modes_still_apply(override, expected):
    evaluator = create_trajectory_match_evaluator(
        tool_args_match_overrides={"get_weather": override},
    )
    result = evaluator(
        outputs=_trajectory("NYC"),
        reference_outputs=_trajectory("SF"),
    )
    assert result["score"] is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trajectory_match_mode", ["strict", "unordered", "subset", "superset"]
)
async def test_unknown_tool_args_override_mode_raises_async(trajectory_match_mode):
    evaluator = create_async_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        tool_args_match_overrides={"get_weather": "strict"},
    )
    with pytest.raises(ValueError, match="Invalid tool args match mode"):
        await evaluator(
            outputs=_trajectory("NYC"),
            reference_outputs=_trajectory("SF"),
        )
