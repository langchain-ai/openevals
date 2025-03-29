import os

import pytest
from e2b_code_interpreter import AsyncSandbox

from openevals.code.e2b.pyright import create_async_e2b_pyright_evaluator


@pytest.mark.skipif(
    os.environ.get("E2B_API_KEY") is None,
    reason="E2B_API_KEY is not set",
)
@pytest.mark.asyncio
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

graph.invoke({})
""",
            True,
        ),
    ],
)
async def test_e2b_pyright_evaluator(inputs, outputs, expected_result):
    sandbox = await AsyncSandbox.create("OpenEvalsPython")
    evaluator = create_async_e2b_pyright_evaluator(
        sandbox=sandbox,
    )
    eval_result = await evaluator(inputs=inputs, outputs=outputs)
    assert eval_result["score"] == expected_result
