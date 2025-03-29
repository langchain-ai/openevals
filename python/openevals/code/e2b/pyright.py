import json

from e2b_code_interpreter import Sandbox, AsyncSandbox, CommandExitException
from langchain_core.language_models.chat_models import BaseChatModel

from typing import Callable, Any, Literal, Optional

from openevals.code.base import (
    _create_base_code_evaluator,
    _create_async_base_code_evaluator,
)
from openevals.types import SimpleEvaluator, SimpleAsyncEvaluator

from openevals.code.e2b.sandbox.files import (
    PYTHON_EVALUATOR_FILE,
    EXTRACT_IMPORT_NAMES,
    PYTHON_EVALUATOR_SEPARATOR,
)

E2B_COMMAND = (" && ").join(
    [
        f"echo '{EXTRACT_IMPORT_NAMES}' > extract_import_names.py",
        f"echo '{PYTHON_EVALUATOR_FILE}' > run_pyright.py",
        "export PIP_DISABLE_PIP_VERSION_CHECK=1",
        "python3 extract_import_names.py > openevals_requirements.txt",
        'if command -v "uv" >/dev/null 2>&1; then uv venv --quiet && uv pip install -r openevals_requirements.txt --quiet; else pip install -r openevals_requirements.txt --quiet --upgrade-strategy only-if-needed; fi',
        'if command -v "pyright" >/dev/null 2>&1; then : ; else npm i -g pyright; fi',
        f"echo '{PYTHON_EVALUATOR_SEPARATOR}'",
        'if command -v "uv" >/dev/null 2>&1; then uv run python run_pyright.py; else python run_pyright.py; fi',
    ]
)


def create_e2b_pyright_evaluator(
    *,
    sandbox: Sandbox,
    sandbox_project_directory: Optional[str] = None,
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
) -> SimpleEvaluator:
    """Creates an evaluator that sets up an E2B sandbox, installs dependencies,
    and uses Pyright to check Python code for type errors.

    Args:
        sandbox (Sandbox): The E2B sandbox environment for code execution.
        sandbox_project_directory (Optional[str], optional): Directory where the code will be evaluated.
            Defaults to "openevals".
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

    def _scorer(outputs: str, **kwargs: Any):
        cwd = sandbox_project_directory or "openevals"
        sandbox.files.write(f"{cwd}/outputs.py", outputs)
        try:
            cmd = sandbox.commands.run(cmd=E2B_COMMAND, cwd=cwd)
            if PYTHON_EVALUATOR_SEPARATOR in cmd.stdout:
                parsed_result = json.loads(
                    cmd.stdout.split(PYTHON_EVALUATOR_SEPARATOR)[1]
                )
                return (parsed_result[0], parsed_result[1])
            return False, cmd.stdout
        except CommandExitException as e:
            return False, str(e)

    return _create_base_code_evaluator(
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=model,
        client=client,
        run_name="e2b_pyright_evaluator",
        feedback_key="pyright_succeeded",
    )


def create_async_e2b_pyright_evaluator(
    *,
    sandbox: AsyncSandbox,
    sandbox_project_directory: Optional[str] = None,
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
) -> SimpleAsyncEvaluator:
    """Creates an asynchronous evaluator that sets up an E2B sandbox, installs dependencies,
    and uses Pyright to check Python code for type errors.

    Args:
        sandbox (AsyncSandbox): The E2B async sandbox environment for code execution.
        sandbox_project_directory (Optional[str], optional): Directory where the code will be evaluated.
            Defaults to "openevals".
        code_extraction_strategy (Literal["none", "llm", "markdown_code_blocks"], optional): Strategy for
            extracting code from the input. Defaults to "none".
        code_extractor (Optional[Callable[[Any], str]], optional): Custom function to extract code from input.
            Only used if code_extraction_strategy is "none".
        model (Optional[str], optional): The model identifier for LLM-based code extraction.
            Only used if code_extraction_strategy is "llm".
        client (Optional[BaseChatModel], optional): The chat model client for LLM-based code extraction.
            Only used if code_extraction_strategy is "llm".

    Returns:
        SimpleAsyncEvaluator: An asynchronous evaluator function that returns a tuple of
            (success: bool, feedback: str) where success indicates if the code passed Pyright's
            type checking and feedback contains error messages or confirmation of success.

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
            cmd = await sandbox.commands.run(cmd=E2B_COMMAND, cwd=cwd)
            if PYTHON_EVALUATOR_SEPARATOR in cmd.stdout:
                parsed_result = json.loads(
                    cmd.stdout.split(PYTHON_EVALUATOR_SEPARATOR)[1]
                )
                return (parsed_result[0], parsed_result[1])
            return False, cmd.stdout
        except CommandExitException as e:
            return False, str(e)

    return _create_async_base_code_evaluator(
        scorer=_scorer,
        code_extraction_strategy=code_extraction_strategy,
        code_extractor=code_extractor,
        model=model,
        client=client,
        run_name="e2b_pyright_evaluator",
        feedback_key="pyright_succeeded",
    )
