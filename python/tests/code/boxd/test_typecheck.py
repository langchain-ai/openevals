import os

import pytest

from openevals.code.boxd.pyright import create_boxd_pyright_evaluator


@pytest.mark.skipif(
    os.environ.get("BOXD_API_KEY") is None,
    reason="BOXD_API_KEY is not set",
)
@pytest.mark.langsmith(output_keys=["outputs"])
@pytest.mark.parametrize(
    "inputs, outputs, expected_result",
    [
        (
            "Generate an app with a bad import.",
            """
from tyjfieowjiofewjiooiing import foo

foo()
""",
            False,
        ),
        (
            "Generate a bad LangGraph app.",
            """
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)
builder.add_node("start", lambda state: state)
builder.compile()

builder.invoke({})
""",
            False,
        ),
        (
            "Generate a good LangGraph app.",
            """
import json
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)
builder.add_node("start", lambda state: state)
builder.add_edge(START, "start")
graph = builder.compile()

graph.invoke({"messages": []})
""",
            True,
        ),
    ],
)
def test_boxd_pyright_evaluator(
    inputs, outputs, expected_result, boxd_client, boxd_machine_id
):
    evaluator = create_boxd_pyright_evaluator(
        boxd_client=boxd_client,
        machine_id=boxd_machine_id,
    )
    eval_result = evaluator(inputs=inputs, outputs=outputs)
    assert eval_result["score"] == expected_result
