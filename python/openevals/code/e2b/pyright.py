import json

from e2b_code_interpreter import Sandbox, CommandExitException
from langchain_core.language_models.chat_models import BaseChatModel

from uuid import uuid4
from typing import Callable, Any, Literal, Optional

from openevals.code.base import _create_base_code_evaluator
from openevals.types import SimpleEvaluator

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
        "python3 extract_import_names.py > requirements.txt",
        'if command -v "uv" >/dev/null 2>&1; then uv venv --quiet && uv pip install -r requirements.txt --quiet; else pip install -r requirements.txt --quiet --upgrade-strategy only-if-needed; fi',
        'if command -v "pyright" >/dev/null 2>&1; then : ; else npm i -g pyright; fi',
        f"echo '{PYTHON_EVALUATOR_SEPARATOR}'",
        'if command -v "uv" >/dev/null 2>&1; then uv run python run_pyright.py; else python run_pyright.py; fi',
    ]
)


def create_e2b_pyright_evaluator(
    *,
    sandbox: Sandbox,
    new_project_directory_per_execution: bool = False,
    code_extraction_strategy: Literal["none", "llm", "markdown_code_blocks"] = "none",
    code_extractor: Optional[Callable[[Any], str]] = None,
    model: Optional[str] = None,
    client: Optional[BaseChatModel] = None,
) -> SimpleEvaluator:
    if code_extraction_strategy != "none" and (model or client):
        raise ValueError(
            "model and client may only be passed if code_extraction_strategy is 'none'"
        )

    def _scorer(outputs: str, **kwargs: Any):
        cwd = new_project_directory_per_execution and uuid4() or "openevals"
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
