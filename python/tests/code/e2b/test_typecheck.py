import pytest
from e2b_code_interpreter import Sandbox

from openevals.code.e2b.pyright import create_e2b_pyright_evaluator


@pytest.fixture(scope="module")
def sandbox():
    sandbox = Sandbox("OpenEvalsPython")
    yield sandbox


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
def test_e2b_pyright_evaluator(inputs, outputs, expected_result, sandbox):
    evaluator = create_e2b_pyright_evaluator(
        sandbox=sandbox,
    )
    eval_result = evaluator(inputs=inputs, outputs=outputs)
    assert eval_result["score"] == expected_result
