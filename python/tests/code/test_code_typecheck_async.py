import pytest

from openevals.code.pyright import create_async_pyright_evaluator
from openevals.code.mypy import create_async_mypy_evaluator

CODE_EXAMPLES = [
    (
        "Sure! Here's a function that returns the sum of two numbers: def sum_of_two_numbers(a, b): return a + b",
        False,
        "all",
    ),
    ("def sum_of_two_numbers(a, b): return a + b", True, "all"),
    (
        """
from fastapi import FastAPI

app = FastAPI()

def read_root():
    return {"Hello": "World"}
        """,
        True,
        "all",
    ),
    (
        """
from fastapi import FastAPIde

app = FastAPI()

def read_root():
    return {"Hello": "World"}
        """,
        False,
        "all",
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
        "all",
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
        "all",
    ),
    (
        """
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    nonlocal nonlocal nonlocal
    return {"Hello": "World"}
```
        """,
        False,
        "markdown_code_blocks",
    ),
    (
        """
Sure! I'll help you write a FastAPI server. Initialize it like this:

```python
from fastapi import FastAPI

app = FastAPI()

```

Then, you can add a route like this:

```
@app.get("/")
def read_root():
    return {"Hello": "World"}
```
        """,
        True,
        "markdown_code_blocks",
    ),
]


@pytest.mark.asyncio
@pytest.mark.langsmith(output_keys=["outputs"])
@pytest.mark.parametrize("outputs, expected_result, strategy", CODE_EXAMPLES)
async def test_pyright_extraction(outputs, expected_result, strategy):
    pyright_evaluator = create_async_pyright_evaluator(
        code_extraction_strategy=strategy
    )
    eval_result = await pyright_evaluator(outputs=outputs)
    assert eval_result["score"] == expected_result


@pytest.mark.asyncio
@pytest.mark.langsmith(output_keys=["outputs"])
@pytest.mark.parametrize("outputs, expected_result, strategy", CODE_EXAMPLES)
async def test_mypy_extraction(outputs, expected_result, strategy):
    mypy_evaluator = create_async_mypy_evaluator(code_extraction_strategy=strategy)
    eval_result = await mypy_evaluator(outputs=outputs)
    assert eval_result["score"] == expected_result


@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_pyright_extraction_llm():
    pyright_evaluator = create_async_pyright_evaluator(
        code_extraction_strategy="llm", model="openai:o4-mini"
    )
    eval_result = await pyright_evaluator(
        outputs="Sure! Here's a function that returns the sum of two numbers: def sum_of_two_numbers(a, b): return a + b"
    )
    assert eval_result["score"]


@pytest.mark.asyncio
@pytest.mark.langsmith
async def test_pyright_extraction_llm_no_code():
    pyright_evaluator = create_async_pyright_evaluator(
        code_extraction_strategy="llm",
        model="openai:o4-mini",
    )
    eval_result = await pyright_evaluator(outputs="I'm doing well, how about you?")
    assert not eval_result["score"]
    assert eval_result["metadata"]["code_extraction_failed"]
