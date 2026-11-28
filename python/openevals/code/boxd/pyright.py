import json

from boxd import AsyncBoxd, Boxd, ExecResult
from langchain_core.language_models.chat_models import BaseChatModel

from typing import Callable, Any, Literal, Optional

from openevals.code.base import (
    _create_base_code_evaluator,
    _create_async_base_code_evaluator,
)
from openevals.types import SimpleEvaluator, SimpleAsyncEvaluator
from openevals.code.boxd.machine import (
    DEFAULT_TIMEOUT_SECONDS,
    _arun_in_machine,
    _run_in_machine,
    _text,
    _validate_timeout,
)

from openevals.code.e2b.sandbox.files import (
    PYTHON_EVALUATOR_FILE,
    EXTRACT_IMPORT_NAMES,
    PYTHON_EVALUATOR_SEPARATOR,
)

BOXD_COMMAND = (" && ").join(
    [
        "export PIP_DISABLE_PIP_VERSION_CHECK=1",
        "python3 extract_import_names.py > openevals_requirements.txt",
        'if command -v "uv" >/dev/null 2>&1; then uv venv --quiet --allow-existing && uv pip install -r openevals_requirements.txt --quiet; else pip install -r openevals_requirements.txt --quiet --upgrade-strategy only-if-needed; fi',
        'export PATH="$(npm config get prefix)/bin:$PATH"',
        'if command -v "pyright" >/dev/null 2>&1; then : ; else npm i -g pyright; fi',
        f"echo '{PYTHON_EVALUATOR_SEPARATOR}'",
        'if command -v "uv" >/dev/null 2>&1; then uv run python run_pyright.py; else python run_pyright.py; fi',
    ]
)


def _project_files(outputs: str) -> dict[str, str]:
    return {
        "outputs.py": outputs,
        "extract_import_names.py": EXTRACT_IMPORT_NAMES,
        "run_pyright.py": PYTHON_EVALUATOR_FILE,
    }


def _pyright_result(result: ExecResult) -> tuple[bool, Optional[str]]:
    stdout = _text(result.stdout)
    if result.exit_code != 0:
        output = _text(result.stderr) or stdout
        return (
            False,
            f"Command exited with code {result.exit_code} and error:\n{output}",
        )
    if PYTHON_EVALUATOR_SEPARATOR in stdout:
        parsed_result = json.loads(stdout.split(PYTHON_EVALUATOR_SEPARATOR)[1])
        return (parsed_result[0], parsed_result[1])
    return False, stdout


def create_boxd_pyright_evaluator(
    *,
    boxd_client: Boxd,
    machine_id: str,
    sandbox_project_directory: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
) -> SimpleEvaluator:
    """Creates an evaluator that installs dependencies in a boxd machine and uses Pyright
    to check Python code for type errors.

    Args:
        boxd_client (Boxd): The boxd client used to reach the machine.
        machine_id (str): The id of the boxd machine to run the type check in.
        sandbox_project_directory (Optional[str], optional): Directory where the code will be evaluated,
            relative to the home directory or absolute. Defaults to "openevals".
        timeout (Optional[float], optional): Seconds to wait for the install and type check. When it
            elapses, the command is stopped and the boxd client raises an error.
            None means no limit. Defaults to 300.
        code_extraction_strategy (Literal["none", "llm", "markdown_code_blocks"], optional): Strategy for
            extracting code from the input. Defaults to "none".
        code_extractor (Optional[Callable[[Any], str]], optional): Custom function to extract code from input.
            Only used if code_extraction_strategy is "none".
        model (Optional[str], optional): The model identifier for LLM-based code extraction.
            Only used if code_extraction_strategy is "llm".
        client (Optional[BaseChatModel], optional): The chat model client for LLM-based code extraction.
            Only used if code_extraction_strategy is "llm".

    Returns:
        SimpleEvaluator: An evaluator function that returns a tuple of
            (success: bool, feedback: str) where success indicates if the code passed Pyright's
            type checking and feedback contains error messages or confirmation of success.

    Raises:
        ValueError: If model or client are provided when code_extraction_strategy is not "llm".
    """
    if code_extraction_strategy != "llm" and (model or client):
        raise ValueError(
            "model and client may only be passed if code_extraction_strategy is 'llm'"
        )
    _validate_timeout(timeout)

    def _scorer(outputs: str, **kwargs: Any):
        cwd = sandbox_project_directory or "openevals"
        result = _run_in_machine(
            boxd_client,
            machine_id,
            cwd=cwd,
            files=_project_files(outputs),
            command=BOXD_COMMAND,
            env=None,
            timeout=timeout,
        )
        return _pyright_result(result)

    return _create_base_code_evaluator(
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=model,
        client=client,
        run_name="boxd_pyright_evaluator",
        feedback_key="pyright_succeeded",
    )


def create_async_boxd_pyright_evaluator(
    *,
    boxd_client: AsyncBoxd,
    machine_id: str,
    sandbox_project_directory: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
) -> SimpleAsyncEvaluator:
    """Creates an async evaluator that installs dependencies in a boxd machine and uses Pyright
    to check Python code for type errors.

    Args:
        boxd_client (AsyncBoxd): The async boxd client used to reach the machine.
        machine_id (str): The id of the boxd machine to run the type check in.
        sandbox_project_directory (Optional[str], optional): Directory where the code will be evaluated,
            relative to the home directory or absolute. Defaults to "openevals".
        timeout (Optional[float], optional): Seconds to wait for the install and type check. When it
            elapses, the command is stopped and the boxd client raises an error.
            None means no limit. Defaults to 300.
        code_extraction_strategy (Literal["none", "llm", "markdown_code_blocks"], optional): Strategy for
            extracting code from the input. Defaults to "none".
        code_extractor (Optional[Callable[[Any], str]], optional): Custom function to extract code from input.
            Only used if code_extraction_strategy is "none".
        model (Optional[str], optional): The model identifier for LLM-based code extraction.
            Only used if code_extraction_strategy is "llm".
        client (Optional[BaseChatModel], optional): The chat model client for LLM-based code extraction.
            Only used if code_extraction_strategy is "llm".

    Returns:
        SimpleAsyncEvaluator: An async evaluator function that returns a tuple of
            (success: bool, feedback: str) where success indicates if the code passed Pyright's
            type checking and feedback contains error messages or confirmation of success.

    Raises:
        ValueError: If model or client are provided when code_extraction_strategy is not "llm".
    """
    if code_extraction_strategy != "llm" and (model or client):
        raise ValueError(
            "model and client may only be passed if code_extraction_strategy is 'llm'"
        )
    _validate_timeout(timeout)

    async def _scorer(outputs: str, **kwargs: Any):
        cwd = sandbox_project_directory or "openevals"
        result = await _arun_in_machine(
            boxd_client,
            machine_id,
            cwd=cwd,
            files=_project_files(outputs),
            command=BOXD_COMMAND,
            env=None,
            timeout=timeout,
        )
        return _pyright_result(result)

    return _create_async_base_code_evaluator(
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=model,
        client=client,
        run_name="boxd_pyright_evaluator",
        feedback_key="pyright_succeeded",
    )
