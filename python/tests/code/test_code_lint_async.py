import pytest

from openevals.code.pyright import create_async_pyright_evaluator
from openevals.code.mypy import create_async_mypy_evaluator

CODE_EXAMPLES = [
    (
        "Sure! Here's a function that returns the sum of two numbers: def sum_of_two_numbers(a, b): return a + b",
        False,
    ),
    ("def sum_of_two_numbers(a, b): return a + b", True),
    (
        """
from fastapi import FastAPI

app = FastAPI()

def read_root():
    return {"Hello": "World"}
        """,
        True,
    ),
    (
        """
from fastapi import FastAPIde

app = FastAPI()

def read_root():
    return {"Hello": "World"}
        """,
        False,
    ),
    (
        """
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
        """,
        True,
    ),
    (
        """
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    nonlocal nonlocal nonlocal
    return {"Hello": "World"}
        """,
        False,
    ),
]


@pytest.mark.asyncio
@pytest.mark.langsmith(output_keys=["outputs"])
@pytest.mark.parametrize("outputs, expected_result", CODE_EXAMPLES)
async def test_pyright_extraction_strategy_default(outputs, expected_result):
    pyright_evaluator = create_async_pyright_evaluator()
    eval_result = await pyright_evaluator(outputs=outputs)
    assert eval_result["score"] == expected_result


@pytest.mark.asyncio
@pytest.mark.langsmith(output_keys=["outputs"])
@pytest.mark.parametrize("outputs, expected_result", CODE_EXAMPLES)
async def test_mypy_extraction_strategy_default(outputs, expected_result):
    mypy_evaluator = create_async_mypy_evaluator()
    eval_result = await mypy_evaluator(outputs=outputs)
    assert eval_result["score"] == expected_result
