import json

import pytest

from openevals.trajectory.match import create_trajectory_match_evaluator


def _trajectory(args: dict) -> list[dict]:
    return [
        {"role": "user", "content": "What time is my flight?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "1",
                    "function": {
                        "name": "get_flight_info",
                        "arguments": json.dumps(args),
                    },
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    "trajectory_match_mode", ["strict", "unordered", "subset", "superset"]
)
def test_custom_matcher_receives_output_args_first(trajectory_match_mode):
    calls: list[tuple[dict, dict]] = []

    def matcher(output_args: dict, reference_args: dict) -> bool:
        calls.append((output_args, reference_args))
        return True

    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        tool_args_match_overrides={"get_flight_info": matcher},
    )
    result = evaluator(
        outputs=_trajectory({"source": "output"}),
        reference_outputs=_trajectory({"source": "reference"}),
    )

    assert result["score"] is True
    assert calls
    assert all(
        output_args == {"source": "output"}
        and reference_args == {"source": "reference"}
        for output_args, reference_args in calls
    )


@pytest.mark.parametrize("trajectory_match_mode", ["unordered", "subset", "superset"])
def test_subset_tool_args_mode_is_consistent_across_trajectory_modes(
    trajectory_match_mode,
):
    # The output args are a subset of the reference args.
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode=trajectory_match_mode,
        tool_args_match_mode="subset",
    )
    result = evaluator(
        outputs=_trajectory({"flight_no": "LX0112"}),
        reference_outputs=_trajectory({"flight_no": "LX0112", "foo": "bar"}),
    )

    assert result["score"] is True
