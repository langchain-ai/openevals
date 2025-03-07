import asyncio
import json
import subprocess
import tempfile
import os

from typing import Optional, Callable, Any, Union, Literal, Awaitable, Tuple

from openevals.code.llm import (
    _create_base_code_evaluator,
    _create_async_base_code_evaluator,
)
from openevals.types import ModelClient, SimpleEvaluator, SimpleAsyncEvaluator

from langchain_core.language_models.chat_models import BaseChatModel


def _parse_mypy_output(stdout: bytes) -> Tuple[bool, str]:
    try:
        # Parse the output - mypy gives line-by-line error messages
        errors = []
        for line in stdout.decode().split("\n"):
            line_components = line.strip().split(":")
            if len(line_components) > 2 and line_components[0].endswith(".py"):
                display_line = ":".join(["python_file.py", *line_components])
                errors.append(display_line)
        score = len(errors) == 0
        return (score, "\n".join(errors))
    except json.JSONDecodeError:
        return (False, f"Failed to parse Mypy output: {stdout.decode()}")


def _analyze_with_mypy(
    *,
    output: str,
    mypy_cli_args: list[str],
) -> Tuple[bool, str]:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp:
        temp.write(output)
        temp_path = temp.name

    try:
        # Run mypy with specific flags
        result = subprocess.run(
            [
                "mypy",
                *mypy_cli_args,
                temp_path,
            ],
            capture_output=True,
        )
        return _parse_mypy_output(result.stdout)  # type: ignore

    finally:
        # Clean up
        os.unlink(temp_path)


async def _analyze_with_mypy_async(
    *,
    output: str,
    mypy_cli_args: list[str],
) -> Tuple[bool, str]:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp:
        temp.write(output)
        temp_path = temp.name

    try:
        # Run mypy with specific flags asynchronously
        process = await asyncio.create_subprocess_exec(
            "mypy",
            *(mypy_cli_args or []),
            temp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()

        return _parse_mypy_output(stdout)  # type: ignore

    finally:
        # Clean up
        os.unlink(temp_path)


def create_mypy_evaluator(
    *,
    mypy_cli_args: list[str] = [
        "--no-incremental",
        "--disallow-untyped-calls",
        "--disallow-incomplete-defs",
        "--ignore-missing-imports",
    ],
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    client: Optional[Union[ModelClient, BaseChatModel]] = None,
    model: Optional[str] = None,
) -> SimpleEvaluator:
    def _scorer(
        *,
        outputs: str,
        **kwargs,
    ):
        return _analyze_with_mypy(
            output=outputs,
            mypy_cli_args=mypy_cli_args,
        )

    return _create_base_code_evaluator(
        model=model,
        client=client,
        run_name="code_llm_as_judge",
        feedback_key="mypy_succeeded",
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
    )


def create_async_mypy_evaluator(
    *,
    mypy_cli_args: list[str] = [
        "--no-incremental",
        "--disallow-untyped-calls",
        "--disallow-incomplete-defs",
        "--ignore-missing-imports",
    ],
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], Union[str, Awaitable[str]]]] = None,
    client: Optional[Union[ModelClient, BaseChatModel]] = None,
    model: Optional[str] = None,
) -> SimpleAsyncEvaluator:
    async def _scorer(
        *,
        outputs: str,
        **kwargs,
    ):
        return await _analyze_with_mypy_async(
            output=outputs,
            mypy_cli_args=mypy_cli_args,
        )

    return _create_async_base_code_evaluator(
        model=model,
        client=client,
        run_name="mypy_check",
        feedback_key="mypy_succeeded",
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
    )
