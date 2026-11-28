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
    EXTRACT_IMPORT_NAMES,
    PYTHON_EVALUATOR_SEPARATOR,
)


def _create_boxd_execution_command(
    *,
    execution_command: str = "python",
) -> str:
    return (" && ").join(
        [
            "export PIP_DISABLE_PIP_VERSION_CHECK=1",
            "python3 extract_import_names.py > openevals_requirements.txt",
            'if command -v "uv" >/dev/null 2>&1; then uv venv --quiet --allow-existing && uv pip install -r openevals_requirements.txt --quiet; else pip install -r openevals_requirements.txt --quiet --upgrade-strategy only-if-needed; fi',
            f"echo '{PYTHON_EVALUATOR_SEPARATOR}'",
            f'if command -v "uv" >/dev/null 2>&1; then uv run {execution_command} outputs.py; else {execution_command} outputs.py; fi',
        ]
    )


def _project_files(outputs: str) -> dict[str, str]:
    return {
        "outputs.py": outputs,
        "extract_import_names.py": EXTRACT_IMPORT_NAMES,
    }


def _execution_result(result: ExecResult) -> tuple[bool, Optional[str]]:
    if result.exit_code == 0:
        return True, None
    output = _text(result.stderr) or _text(result.stdout)
    return (
        False,
        f"Command exited with code {result.exit_code} and error:\n{output}",
    )


def create_boxd_execution_evaluator(
    *,
    boxd_client: Boxd,
    machine_id: str,
    environment_variables: Optional[dict[str, str]] = None,
    execution_command: str = "python",
    sandbox_project_directory: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
) -> SimpleEvaluator:
    """Creates an evaluator that executes code in a boxd machine and returns whether
    execution was successful.

    Args:
        boxd_client (Boxd): The boxd client used to reach the machine.
        machine_id (str): The id of the boxd machine to run the code in.
        environment_variables (Optional[dict[str, str]], optional): Environment variables to set
            for the command in the machine. Defaults to None.
        execution_command (Optional[str], optional): Command used to execute the code.
            Defaults to "python".
        sandbox_project_directory (Optional[str], optional): Directory where the code will be executed,
            relative to the home directory or absolute. Defaults to "openevals".
        timeout (Optional[float], optional): Seconds to wait for the install and run. When it elapses,
            the command is stopped and the boxd client raises an error.
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
        SimpleEvaluator: An evaluator function that returns a tuple of (success: bool, feedback: str)
            where success indicates if the code executed without errors and feedback contains the
            execution output or error messages.

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
            command=_create_boxd_execution_command(execution_command=execution_command),
            env=environment_variables,
            timeout=timeout,
        )
        return _execution_result(result)

    return _create_base_code_evaluator(
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=model,
        client=client,
        run_name="boxd_execution_evaluator",
        feedback_key="execution_succeeded",
    )


def create_async_boxd_execution_evaluator(
    *,
    boxd_client: AsyncBoxd,
    machine_id: str,
    environment_variables: Optional[dict[str, str]] = None,
    execution_command: str = "python",
    sandbox_project_directory: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
) -> SimpleAsyncEvaluator:
    """Creates an async evaluator that executes code in a boxd machine and returns whether
    execution was successful.

    Args:
        boxd_client (AsyncBoxd): The async boxd client used to reach the machine.
        machine_id (str): The id of the boxd machine to run the code in.
        environment_variables (Optional[dict[str, str]], optional): Environment variables to set
            for the command in the machine. Defaults to None.
        execution_command (Optional[str], optional): Command used to execute the code.
            Defaults to "python".
        sandbox_project_directory (Optional[str], optional): Directory where the code will be executed,
            relative to the home directory or absolute. Defaults to "openevals".
        timeout (Optional[float], optional): Seconds to wait for the install and run. When it elapses,
            the command is stopped and the boxd client raises an error.
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
        SimpleAsyncEvaluator: An evaluator function that returns a tuple of (success: bool, feedback: str)
            where success indicates if the code executed without errors and feedback contains the
            execution output or error messages.

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
            command=_create_boxd_execution_command(execution_command=execution_command),
            env=environment_variables,
            timeout=timeout,
        )
        return _execution_result(result)

    return _create_async_base_code_evaluator(
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=model,
        client=client,
        run_name="boxd_execution_evaluator",
        feedback_key="execution_succeeded",
    )
