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


def _parse_pyright_output(stdout: bytes) -> Tuple[bool, str]:
    try:
        # Parse the JSON output
        output = json.loads(stdout)

        errors = []
        for error in output.get("generalDiagnostics", []):
            if (
                error.get("severity", None) == "error"
                and error.get("rule", None) != "reportMissingImports"
            ):
                del error["file"]
                errors.append(error)
        score = len(errors) == 0
        return (score, json.dumps(errors))
    except json.JSONDecodeError:
        print(stdout.decode())
        return (False, f"Failed to parse Pyright output: {stdout.decode()}")


def _analyze_with_pyright(
    *,
    output: str,
    pyright_cli_args: list[str],
):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp:
        temp.write(output)
        temp_path = temp.name

    try:
        result = subprocess.run(
            [
                "pyright",
                "--outputjson",
                "--level",
                "error",  # Only report errors, not warnings
                *(pyright_cli_args or []),
                temp_path,
            ],
            capture_output=True,
        )

        return _parse_pyright_output(result.stdout)  # type: ignore

    finally:
        # Clean up
        os.unlink(temp_path)


async def _analyze_with_pyright_async(
    *,
    output: str,
    pyright_cli_args: list[str],
):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp:
        temp.write(output)
        temp_path = temp.name

    try:
        # Use asyncio.create_subprocess_exec for async subprocess execution
        import asyncio

        process = await asyncio.create_subprocess_exec(
            "pyright",
            "--outputjson",
            "--level",
            "error",  # Only report errors, not warnings
            *(pyright_cli_args or []),
            temp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await process.communicate()
        return _parse_pyright_output(stdout)  # type: ignore

    finally:
        # Clean up
        os.unlink(temp_path)


def create_pyright_evaluator(
    *,
    pyright_cli_args: list[str] = [],
    code_extraction_strategy: Literal["all", "llm"] = "all",
    code_extractor: Optional[Callable[[Any], str]] = None,
    client: Optional[Union[ModelClient, BaseChatModel]] = None,
    model: Optional[str] = None,
) -> SimpleEvaluator:
    def _scorer(
        *,
        outputs: str,
        **kwargs,
    ):
        return _analyze_with_pyright(
            output=outputs,
            pyright_cli_args=pyright_cli_args,
        )

    return _create_base_code_evaluator(
        model=model,
        client=client,
        run_name="code_llm_as_judge",
        feedback_key="pyright_succeeded",
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
    )


def create_async_pyright_evaluator(
    *,
    pyright_cli_args: list[str] = [],
    code_extraction_strategy: Literal["all", "llm"] = "all",
    code_extractor: Optional[Callable[[Any], Union[str, Awaitable[str]]]] = None,
    client: Optional[Union[ModelClient, BaseChatModel]] = None,
    model: Optional[str] = None,
) -> SimpleAsyncEvaluator:
    async def _scorer(
        *,
        outputs: str,
        **kwargs,
    ):
        return await _analyze_with_pyright_async(
            output=outputs,
            pyright_cli_args=pyright_cli_args,
        )

    return _create_async_base_code_evaluator(
        model=model,
        client=client,
        run_name="code_llm_as_judge",
        feedback_key="pyright_succeeded",
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
    )
