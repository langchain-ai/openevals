import json

import pytest

from openevals.trajectory.match import (
    create_async_trajectory_match_evaluator,
    create_trajectory_match_evaluator,
)


def _tool_call(args: dict, name: str = "search") -> dict:
    return {"function": {"name": name, "arguments": json.dumps(args)}}


def _trajectory(*tool_calls: dict) -> list[dict]:
    return [
        {"role": "user", "content": "Find the flights."},
        {"role": "assistant", "content": "", "tool_calls": list(tool_calls)},
    ]


NARROW = _tool_call({"origin": "SFO"})
BROAD = _tool_call({"origin": "SFO", "destination": "JFK"})


@pytest.mark.parametrize(
    "trajectory_match_mode", ["strict", "unordered", "subset", "superset"]
)
@pytest.mark.parametrize(
    "tool_args_match_mode, outputs, reference_outputs",
    [
        ("superset", _trajectory(BROAD, NARROW), _trajectory(NARROW, BROAD)),
        ("subset", _trajectory(NARROW, BROAD), _trajectory(BROAD, NARROW)),
    ],
)
def test_order_of_tool_calls_does_not_decide_the_match(
    trajectory_match_mode, tool_args_match_mode, outputs, reference_outputs
):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        tool_args_match_mode=tool_args_match_mode,
    )
    result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert result["score"] is True


@pytest.mark.parametrize(
    "trajectory_match_mode", ["strict", "unordered", "subset", "superset"]
)
@pytest.mark.asyncio
async def test_order_of_tool_calls_does_not_decide_the_match_async(
    trajectory_match_mode,
):
    evaluator = create_async_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        tool_args_match_mode="superset",
    )
    result = await evaluator(
        outputs=_trajectory(BROAD, NARROW),
        reference_outputs=_trajectory(NARROW, BROAD),
    )
    assert result["score"] is True


@pytest.mark.parametrize("trajectory_match_mode", ["strict", "unordered", "superset"])
def test_one_output_call_cannot_satisfy_two_reference_calls(trajectory_match_mode):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        tool_args_match_mode="superset",
    )
    result = evaluator(
        outputs=_trajectory(BROAD, _tool_call({"origin": "LAX"})),
        reference_outputs=_trajectory(NARROW, NARROW),
    )
    assert result["score"] is False


def test_matcher_override_uses_the_assignment():
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="unordered",
        tool_args_match_overrides={
            "search": lambda output, reference: reference.items() <= output.items()
        },
    )
    result = evaluator(
        outputs=_trajectory(BROAD, NARROW),
        reference_outputs=_trajectory(NARROW, BROAD),
    )
    assert result["score"] is True
