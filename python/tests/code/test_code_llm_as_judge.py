import pytest

from openevals.code.llm import create_code_llm_as_judge, CODE_CORRECTNESS_PROMPT


@pytest.mark.langsmith(output_keys=["outputs"])
@pytest.mark.parametrize(
    "inputs, outputs, expected_result",
    [
        (
            "Generate a function that returns the sum of two numbers",
            "Sure! Here's a function that returns the sum of two numbers: def sum_of_two_numbers(a, b): return a + b",
            False,
        ),
        (
            "Generate a function that returns the sum of two numbers",
            "def sum_of_two_numbers(a, b): return a + b",
            True,
        ),
        (
            "Generate a working web server in Python with FastAPI.",
            """
        from fastapi import FastAPI
        app = FastAPI()
        def read_root():
            return {"Hello": "World"}""",
            False,
        ),
        (
            "Generate a working web server in Python with FastAPI.",
            """
        from fastapi import FastAPI
        app = FastAPI()
        @app.get("/")
        def read_root():
            return {"Hello": "World"}""",
            True,
        ),
        (
            """
Rewrite the code below to be async:

```python
def _run_mypy(
    *,
    filepath: str,
    mypy_cli_args: list[str],
) -> Tuple[bool, str]:
    result = subprocess.run(
        [
            "mypy",
            *mypy_cli_args,
            filepath,
        ],
        capture_output=True,
    )
    return _parse_mypy_output(result.stdout)
```
""",
            """
```python
async def _run_mypy_async(
    *,
    filepath: str,
    mypy_cli_args: list[str],
) -> Tuple[bool, str]:
    process = await asyncio.create_subprocess_exec(
        "mypy",
        *(mypy_cli_args or []),
        filepath,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()

    return _parse_mypy_output(stdout)
```""",
            True,
        ),
        (
            """
Rewrite the code below to be async:

```python
def _run_mypy(
    *,
    filepath: str,
    mypy_cli_args: list[str],
) -> Tuple[bool, str]:
    result = subprocess.run(
        [
            "mypy",
            *mypy_cli_args,
            filepath,
        ],
        capture_output=True,
    )
    return _parse_mypy_output(result.stdout)
```
""",
            """
```python
async def _run_mypy_async(
    *,
    filepath: str,
    mypy_cli_args: list[str],
) -> Tuple[bool, str]:
    process = await subprocess.run(
        [
            "mypy",
            *mypy_cli_args,
            filepath,
        ],
    )
    stdout, _ = await process.communicate()

    return _parse_mypy_output(stdout)
```""",
            False,
        ),
    ],
)
def test_code_llm_as_judge_extraction_strategy_default(
    inputs, outputs, expected_result
):
    llm_as_judge = create_code_llm_as_judge(
        prompt=CODE_CORRECTNESS_PROMPT,
        model="openai:o3-mini",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    print(eval_result)
    assert eval_result["score"] == expected_result


@pytest.mark.langsmith(output_keys=["outputs"])
@pytest.mark.parametrize(
    "inputs, outputs, expected_result",
    [
        (
            "Generate a function that returns the sum of two numbers",
            "Sure! Here's a function that returns the sum of two numbers: def sum_of_two_numbers(a, b): return a + b",
            True,
        ),
        (
            "Generate a working web server in Python with FastAPI.",
            """
from fastapi import FastAPI

app = FastAPI()

def read_root():
    return {"Hello": "World"}""",
            False,
        ),
        (
            "Generate a working web server in Python with FastAPI.",
            """
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}""",
            True,
        ),
    ],
)
def test_code_llm_as_judge_extraction_strategy_llm(inputs, outputs, expected_result):
    llm_as_judge = create_code_llm_as_judge(
        prompt=CODE_CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        code_extraction_strategy="llm",
    )
    eval_result = llm_as_judge(inputs=inputs, outputs=outputs)
    assert eval_result["score"] == expected_result
