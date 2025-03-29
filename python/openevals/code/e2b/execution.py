from e2b_code_interpreter import Sandbox, CommandExitException, AsyncSandbox
from langchain_core.language_models.chat_models import BaseChatModel

from typing import Callable, Any, Literal, Optional

from openevals.code.base import (
    _create_base_code_evaluator,
    _create_async_base_code_evaluator,
)
from openevals.types import SimpleEvaluator, SimpleAsyncEvaluator

from openevals.code.e2b.sandbox.files import (
    EXTRACT_IMPORT_NAMES,
    PYTHON_EVALUATOR_SEPARATOR,
)


def _create_e2b_execution_command(
    *,
    execution_command: str = "python",
) -> str:
    return (" && ").join(
        [
            f"echo '{EXTRACT_IMPORT_NAMES}' > extract_import_names.py",
            "export PIP_DISABLE_PIP_VERSION_CHECK=1",
            "python3 extract_import_names.py > openevals_requirements.txt",
            'if command -v "uv" >/dev/null 2>&1; then uv venv --quiet && uv pip install -r openevals_requirements.txt --quiet; else pip install -r openevals_requirements.txt --quiet --upgrade-strategy only-if-needed; fi',
            f"echo '{PYTHON_EVALUATOR_SEPARATOR}'",
            f'if command -v "uv" >/dev/null 2>&1; then uv run {execution_command} outputs.py; else {execution_command} outputs.py; fi',
        ]
    )


def create_e2b_execution_evaluator(
    *,
    sandbox: Sandbox,
    environment_variables: Optional[dict[str, str]] = None,
    execution_command: str = "python",
    sandbox_project_directory: Optional[str] = None,
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
) -> SimpleEvaluator:
    """Creates an evaluator that executes code in an E2B sandbox environment and returns whether
    execution was successful.

    Args:
        sandbox (Sandbox): The E2B sandbox environment for code execution.
        environment_variables (Optional[dict[str, str]], optional): Environment variables to set in the sandbox.
            Defaults to None.
        execution_command (Optional[str], optional): Command used to execute the code.
            Defaults to "python".
        sandbox_project_directory (Optional[str], optional): Directory where the code will be executed.
            Defaults to None.
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

    def _scorer(outputs: str, **kwargs: Any):
        cwd = sandbox_project_directory or "openevals"
        sandbox.files.write(f"{cwd}/outputs.py", outputs)
        try:
            cmd = _create_e2b_execution_command(execution_command=execution_command)
            sandbox.commands.run(cmd=cmd, cwd=cwd, envs=environment_variables)
            return True, None
        except CommandExitException as e:
            return False, str(e)

    return _create_base_code_evaluator(
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=model,
        client=client,
        run_name="e2b_execution_evaluator",
        feedback_key="execution_succeeded",
    )


def create_async_e2b_execution_evaluator(
    *,
    sandbox: AsyncSandbox,
    environment_variables: Optional[dict[str, str]] = None,
    execution_command: str = "python",
    sandbox_project_directory: Optional[str] = None,
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
) -> SimpleAsyncEvaluator:
    """Creates an async evaluator that executes code in an E2B sandbox environment and returns whether
    execution was successful.

    Args:
        sandbox (AsyncSandbox): The E2B async sandbox environment for code execution.
        environment_variables (Optional[dict[str, str]], optional): Environment variables to set in the sandbox.
            Defaults to None.
        execution_command (Optional[str], optional): Command used to execute the code.
            Defaults to "python".
        sandbox_project_directory (Optional[str], optional): Directory where the code will be executed.
            Defaults to None.
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

    async def _scorer(outputs: str, **kwargs: Any):
        cwd = sandbox_project_directory or "openevals"
        await sandbox.files.write(f"{cwd}/outputs.py", outputs)
        try:
            cmd = _create_e2b_execution_command(execution_command=execution_command)
            await sandbox.commands.run(cmd=cmd, cwd=cwd, envs=environment_variables)
            return True, None
        except CommandExitException as e:
            return False, str(e)

    return _create_async_base_code_evaluator(
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=model,
        client=client,
        run_name="e2b_execution_evaluator",
        feedback_key="execution_succeeded",
    )
