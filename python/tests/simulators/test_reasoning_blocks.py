from langchain_core.messages import AIMessage

from openevals.simulators import run_multiturn_simulation

REASONING = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "Checking refund policy."}],
}


def _run(app_reply):
    seen = []

    def stopping_condition(trajectory, **kwargs):
        seen.append(trajectory)
        return True

    res = run_multiturn_simulation(
        app=lambda inputs, **kwargs: app_reply,
        user=["Give me a refund!"],
        stopping_condition=stopping_condition,
        max_turns=1,
    )
    return res["trajectory"], seen


def test_reasoning_blocks_are_dropped_from_trajectory():
    reply = AIMessage(content=[REASONING, {"type": "text", "text": "Refund granted."}])
    trajectory, seen = _run(reply)
    for message in (trajectory[-1], seen[0][-1]):
        # The stopping condition receives the same cleaned trajectory.
        assert "reasoning" not in str(message["content"])
        assert "Refund granted." in str(message["content"])


def test_reasoning_only_content_becomes_empty_string():
    reply = AIMessage(
        content=[REASONING],
        tool_calls=[{"name": "give_refund", "args": {}, "id": "call_1"}],
    )
    trajectory, _ = _run(reply)
    assert trajectory[-1]["content"] == ""
    assert trajectory[-1]["tool_calls"][0]["function"]["name"] == "give_refund"


def test_messages_without_reasoning_are_unchanged():
    reply = {"role": "assistant", "content": "Refund granted.", "id": "m1"}
    trajectory, _ = _run(reply)
    assert trajectory[-1] == reply


def test_function_call_blocks_are_dropped_but_tool_calls_kept():
    reply = AIMessage(
        content=[
            REASONING,
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "give_refund",
                "arguments": "{}",
            },
        ],
        tool_calls=[{"name": "give_refund", "args": {}, "id": "call_1"}],
    )
    trajectory, _ = _run(reply)
    assert "function_call" not in str(trajectory[-1]["content"])
    assert trajectory[-1]["tool_calls"][0]["id"] == "call_1"
